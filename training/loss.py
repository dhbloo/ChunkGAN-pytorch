import torch
from torch import nn
from utils.misc import ddp_sync
from utils import training_stats
from utils import conv2d_gradfix


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self, gan_mode, device, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label, device=device))
        self.register_buffer('fake_label', torch.tensor(target_fake_label, device=device))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ChunkGANLoss():
    def __init__(self,
                 device,
                 chunk_size,
                 G,
                 D,
                 Enc_c,
                 Enc_bg,
                 Dec_c,
                 Dec_bg,
                 gp_lambda,
                 EncDec_c_lambda=1,
                 EncDec_bg_lambda=1,
                 G_cycle_lambda=1,
                 G_cycle_chunk_lambda=1,
                 G_disc_lambda=1,
                 G_chunk_lambda=1,
                 D_lambda=1):
        super(ChunkGANLoss, self).__init__()
        self.device = device
        self.chunk_size = chunk_size
        self.G = G
        self.D = D
        self.Enc_c = Enc_c
        self.Enc_bg = Enc_bg
        self.Dec_c = Dec_c
        self.Dec_bg = Dec_bg
        self.gp_lambda = gp_lambda
        self.EncDec_c_lambda = EncDec_c_lambda
        self.EncDec_bg_lambda = EncDec_bg_lambda
        self.G_cycle_lambda = G_cycle_lambda
        self.G_cycle_chunk_lambda = G_cycle_chunk_lambda
        self.G_disc_lambda = G_disc_lambda
        self.G_chunk_lambda = G_chunk_lambda
        self.D_lambda = D_lambda
        self.cycle_loss = nn.L1Loss()
        self.cycle_loss_sum = nn.L1Loss(reduction='sum')
        self.gan_loss = GANLoss('lsgan', device)
        self.latent_cycle_loss = nn.MSELoss()

    def _chunk_shape(self):
        return (self.chunk_size, self.chunk_size)

    def get_mask_image(self, real_image, chunks):
        mask_image = torch.ones((real_image.shape[0], 1, *real_image.shape[2:]),
                                device=self.device)
        for batch_idx in range(len(chunks)):
            for x, y, w, h in chunks[batch_idx]:
                mask_image[batch_idx, :, y:y + h, x:x + w] = 0
        return mask_image

    def accumulate_gradients(self, phase, real_image, chunks, G_inputs, sync, gain):
        assert phase in [
            'EncDec_c_main', 'EncDec_c_both', 'EncDec_bg_main', 'EncDec_bg_both', 'G_main',
            'G_both', 'D_main', 'D_reg', 'D_both'
        ]
        do_EncDec_c_main = phase in ['EncDec_c_main', 'EncDec_c_both']
        do_EncDec_bg_main = phase in ['EncDec_bg_main', 'EncDec_bg_both']
        do_G_main = phase in ['G_main', 'G_both']
        do_G_disc = do_G_main and self.G_disc_lambda != 0
        do_G_chunk = do_G_main and self.G_chunk_lambda != 0
        do_D_main = phase in ['D_main', 'D_both'] and self.D_lambda != 0
        do_D_reg = phase in ['D_reg', 'D_both'] and self.gp_lambda != 0

        # EncDec_c_main: Cycle consistency loss for chunk encoder/generator
        if do_EncDec_c_main:
            loss_EncDec_c = None
            with torch.autograd.profiler.record_function('EncDec_c_forward'):
                losses_EncDec_c = []
                for batch_idx in range(len(chunks)):
                    if len(chunks[batch_idx]) > 0:
                        real_chunks = []
                        for x, y, w, h in chunks[batch_idx]:
                            real_chunk = real_image[batch_idx:batch_idx + 1, :, y:y + h, x:x + w]
                            real_chunk = nn.functional.interpolate(real_chunk,
                                                                   self._chunk_shape(),
                                                                   mode='bilinear',
                                                                   align_corners=False)
                            real_chunks.append(real_chunk)
                        real_chunk = torch.cat(real_chunks)
                        with ddp_sync(self.Enc_c, sync):
                            reak_chunk_z = self.Enc_c(real_chunk)
                        with ddp_sync(self.Dec_c, sync):
                            fake_chunk = self.Dec_c(reak_chunk_z)
                        losses_EncDec_c += [self.cycle_loss(fake_chunk, real_chunk)]
                if len(losses_EncDec_c) > 0:
                    loss_EncDec_c = torch.stack(losses_EncDec_c).mean()
                    training_stats.report('Loss/EncDec_c/loss', loss_EncDec_c)
            if loss_EncDec_c is not None:
                with torch.autograd.profiler.record_function('EncDec_c_backward'):
                    loss_EncDec_c.mul(self.EncDec_c_lambda).mul(gain).backward()

            # with torch.autograd.profiler.record_function('DecEec_c_forward'):
            #     with ddp_sync(self.Dec_c, sync):
            #         fake_chunk = self.Dec_c(latent_c)
            #     with ddp_sync(self.Enc_c, sync):
            #         fake_chunk_z = self.Enc_c(fake_chunk)
            #     loss_DecEec_c = self.cycle_loss(fake_chunk_z, latent_c)
            #     training_stats.report('Loss/DecEnc_c/loss', loss_DecEec_c)
            # with torch.autograd.profiler.record_function('DecEec_c_backward'):
            #     loss_DecEec_c.mul(self.EncDec_c_lambda).mul(gain).backward()

        # EncDec_bg_main: Cycle consistency loss for background encoder/generator
        if do_EncDec_bg_main:
            with torch.autograd.profiler.record_function('EncDec_bg_forward'):
                mask_image = self.get_mask_image(real_image, chunks)
                masked_real_bg = torch.mul(real_image, mask_image)
                with ddp_sync(self.Enc_bg, sync):
                    bg_z = self.Enc_bg(masked_real_bg, mask_image)
                with ddp_sync(self.Dec_bg, sync):
                    fake_bg = self.Dec_bg(bg_z)
                masked_fake_bg = torch.mul(fake_bg, mask_image)
                loss_EncDec_bg = self.cycle_loss(masked_fake_bg, masked_real_bg)
                training_stats.report('Loss/EncDec_bg/loss', loss_EncDec_bg)
            with torch.autograd.profiler.record_function('EncDec_bg_backward'):
                loss_EncDec_bg.mul(self.EncDec_bg_lambda).mul(gain).backward()

            # with torch.autograd.profiler.record_function('DecEec_bg_forward'):
            #     with ddp_sync(self.Dec_bg, sync):
            #         fake_bg = self.Dec_bg(latent_bg)
            #     with ddp_sync(self.Enc_bg, sync):
            #         fake_bg_z = self.Enc_bg(
            #             fake_bg,
            #             torch.ones((fake_bg.shape[0], 1, *fake_bg.shape[2:]), device=self.device))
            #     loss_DecEec_bg = self.cycle_loss(fake_bg_z, latent_bg)
            #     training_stats.report('Loss/DecEnc_bg/loss', loss_DecEec_bg)
            # with torch.autograd.profiler.record_function('DecEec_bg_backward'):
            #     loss_DecEec_bg.mul(self.EncDec_bg_lambda).mul(gain).backward()

        # G_main: Generator loss.
        if do_G_main:
            # Cycle consistency loss for generator
            with torch.autograd.profiler.record_function('Gcycle_forward'):
                # Get chunk latents using chunk encoder
                chunk_latents = []
                with ddp_sync(self.Enc_c, sync=False):
                    for batch_idx in range(len(chunks)):
                        batch_chunk_latents = []
                        for x, y, w, h in chunks[batch_idx]:
                            real_chunk = real_image[batch_idx:batch_idx + 1, :, y:y + h, x:x + w]
                            real_chunk = nn.functional.interpolate(real_chunk,
                                                                   self._chunk_shape(),
                                                                   mode='bilinear',
                                                                   align_corners=False)
                            batch_chunk_latents += [self.Enc_c(real_chunk)]
                        chunk_latents += [
                            torch.cat(batch_chunk_latents)
                            if len(batch_chunk_latents) > 0 else None
                        ]
                # Get background latent using background encoder
                mask_image = self.get_mask_image(real_image, chunks)
                masked_real_bg = torch.mul(real_image, mask_image)
                with ddp_sync(self.Enc_bg, sync=False):
                    bg_latent = self.Enc_bg(masked_real_bg, mask_image)
                # Synthesis final image using generator
                with ddp_sync(self.G, sync=not (do_G_disc or do_G_chunk)):
                    fake_image = self.G(bg_latent, chunk_latents, chunks)
                # Calc image l2 loss
                loss_Gcycle = self.cycle_loss(fake_image, real_image)
                training_stats.report('Loss/Gcycle/loss', loss_Gcycle)

                chunk_mask_image = torch.sub(1, mask_image)
                chunk_mask_image_sum = chunk_mask_image.sum()
                loss_Gcycle_chunk = 0
                if chunk_mask_image_sum > 0:
                    masked_real_chunk = torch.mul(real_image, chunk_mask_image)
                    masked_fake_chunk = torch.mul(fake_image, chunk_mask_image)
                    loss_Gcycle_chunk = self.cycle_loss_sum(
                        masked_fake_chunk, masked_real_chunk).div(chunk_mask_image_sum)
                    training_stats.report('Loss/Gcycle_chunk/loss', loss_Gcycle_chunk)
            with torch.autograd.profiler.record_function('Gcycle_backward'):
                (loss_Gcycle * self.G_cycle_lambda +
                 loss_Gcycle_chunk * self.G_cycle_chunk_lambda).mul(gain).backward()

            # Maximize logits for generated images.
            if do_G_disc:
                with torch.autograd.profiler.record_function('Gdisc_forward'):
                    with ddp_sync(self.G, sync=not do_G_chunk):
                        gen_image = self.G(*G_inputs)
                    with ddp_sync(self.D, sync=False):
                        gen_logits = self.D(gen_image)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    loss_Gdisc = self.gan_loss(gen_logits, True)
                    training_stats.report('Loss/Gdisc/loss', loss_Gdisc)
                with torch.autograd.profiler.record_function('Gdisc_backward'):
                    loss_Gdisc.mul(self.G_disc_lambda).mul(gain).backward()

            # Cycle consistency loss for chunk latents
            if do_G_chunk:
                loss_Gchunk = None
                with torch.autograd.profiler.record_function('Gchunk_forward'):
                    bg_latent, chunk_latents, chunk_trans = G_inputs
                    with ddp_sync(self.G, sync):
                        # Images are generated without background latents
                        gen_image = self.G(bg_latent,
                                           chunk_latents,
                                           chunk_trans,
                                           no_background=True)

                    losses_Gchunk = []
                    for batch_idx in range(len(chunk_trans)):
                        if len(chunk_trans[batch_idx]) > 0:
                            gen_chunks = []
                            for x, y, w, h in chunk_trans[batch_idx]:
                                gen_chunk = gen_image[batch_idx:batch_idx + 1, :, y:y + h, x:x + w]
                                gen_chunk = nn.functional.interpolate(gen_chunk,
                                                                      self._chunk_shape(),
                                                                      mode='bilinear',
                                                                      align_corners=False)
                                gen_chunks.append(gen_chunk)
                            gen_chunk = torch.cat(gen_chunks)
                            with ddp_sync(self.Enc_c, sync=False):
                                gen_chunk_z = self.Enc_c(gen_chunk)
                            losses_Gchunk += [
                                self.latent_cycle_loss(gen_chunk_z, chunk_latents[batch_idx])
                            ]
                    if len(losses_Gchunk) > 0:
                        loss_Gchunk = torch.stack(losses_Gchunk).mean()
                        training_stats.report('Loss/Gchunk/loss', loss_Gchunk)

                if loss_Gchunk is not None:
                    with torch.autograd.profiler.record_function('Gchunk_backward'):
                        (gen_image.mean() * 0 + loss_Gchunk).mul(
                            self.G_chunk_lambda).mul(gain).backward()

        # D_main: Discriminator loss.
        if do_D_main or do_D_reg:
            # Minimize logits for generated images.
            if do_D_main:
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    with ddp_sync(self.G, sync=False):
                        gen_image = self.G(*G_inputs)
                    with ddp_sync(self.D, sync=False):  # Gets synced by loss_Dreal.
                        gen_logits = self.D(gen_image)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    loss_Dgen = self.gan_loss(gen_logits, False)

            with torch.autograd.profiler.record_function('Dreal_Dreg_forward'):
                real_image_tmp = real_image.detach().requires_grad_(do_D_reg)
                with ddp_sync(self.D, sync):
                    real_logits = self.D(real_image_tmp)
                training_stats.report('Loss/scores/real', real_logits)

                # Maximize logits for real images.
                if do_D_main:
                    loss_Dreal = nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    loss_Dreal = self.gan_loss(real_logits, True)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                # Discriminator regularization.
                if do_D_reg:
                    with torch.autograd.profiler.record_function(
                            'D_grads'), conv2d_gradfix.no_weight_gradients():
                        grads = torch.autograd.grad(outputs=[real_logits.sum()],
                                                    inputs=[real_image_tmp],
                                                    create_graph=True,
                                                    only_inputs=True)[0]
                    grads = grads.view(real_image_tmp.size(0), -1)
                    loss_Dreg = ((grads + 1e-16).norm(2, dim=1) - 1.0).square().mean()
                    training_stats.report('Loss/D/grads', grads)
                    training_stats.report('Loss/D/reg', loss_Dreg)

            if do_D_main:
                with torch.autograd.profiler.record_function('Dmain_backward'):
                    (loss_Dgen + loss_Dreal).mul(self.D_lambda).mul(gain).backward()

            if do_D_reg:
                with torch.autograd.profiler.record_function('Dr1_backward'):
                    # (real_logits * 0) added to solve DDP 'use all forward outputs in backward'
                    (real_logits.mean() * 0 + loss_Dreg).mul(self.gp_lambda).mul(gain).backward()
