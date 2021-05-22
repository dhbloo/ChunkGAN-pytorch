import torch
from torch import nn
from utils.misc import ddp_sync
from utils import training_stats
from utils import conv2d_gradfix


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
                 r1_gamma,
                 EncDec_c_lambda=1,
                 EncDec_bg_lambda=1,
                 G_cycle_lambda=1,
                 G_disc_lambda=1,
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
        self.r1_gamma = r1_gamma
        self.EncDec_c_lambda = EncDec_c_lambda
        self.EncDec_bg_lambda = EncDec_bg_lambda
        self.G_cycle_lambda = G_cycle_lambda
        self.G_disc_lambda = G_disc_lambda
        self.D_lambda = D_lambda
        self.cycle_loss = nn.SmoothL1Loss()

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
        do_D_main = phase in ['D_main', 'D_both'] and self.D_lambda != 0
        do_D_r1 = phase in ['D_reg', 'D_both'] and self.r1_gamma != 0

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
                with ddp_sync(self.G, sync=False):  # Gets synced by loss_Gdisc.
                    fake_image = self.G(bg_latent, chunk_latents, chunks)
                # Calc image l2 loss
                loss_Gcycle = self.cycle_loss(fake_image, real_image)
                training_stats.report('Loss/Gcycle/loss', loss_Gcycle)
            with torch.autograd.profiler.record_function('Gcycle_backward'):
                loss_Gcycle.mul(self.G_cycle_lambda).mul(gain).backward()

            # Maximize logits for generated images.
            if self.G_disc_lambda != 0:
                with torch.autograd.profiler.record_function('Gdisc_forward'):
                    with ddp_sync(self.G, sync):
                        gen_image = self.G(*G_inputs)
                    with ddp_sync(self.D, sync=False):
                        gen_logits = self.D(gen_image)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    loss_Gdisc = nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                    training_stats.report('Loss/Gdisc/loss', loss_Gdisc)
                with torch.autograd.profiler.record_function('Gdisc_backward'):
                    loss_Gdisc.mean().mul(self.G_disc_lambda).mul(gain).backward()

        # D_main: Discriminator loss.
        if do_D_main or do_D_r1:
            # Minimize logits for generated images.
            if do_D_main:
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    with ddp_sync(self.G, sync=False):
                        gen_image = self.G(*G_inputs)
                    with ddp_sync(self.D, sync=False):  # Gets synced by loss_Dreal.
                        gen_logits = self.D(gen_image)
                    training_stats.report('Loss/scores/fake', gen_logits)
                    loss_Dgen = nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
                with torch.autograd.profiler.record_function('Dgen_backward'):
                    loss_Dgen.mean().mul(self.D_lambda).mul(gain).backward()

            with torch.autograd.profiler.record_function('Dreal_Dr1_forward'):
                real_image_tmp = real_image.detach().requires_grad_(do_D_r1)
                with ddp_sync(self.D, sync):
                    real_logits = self.D(real_image_tmp)
                training_stats.report('Loss/scores/real', real_logits)

                # Maximize logits for real images.
                if do_D_main:
                    loss_Dreal = nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                # Discriminator regularization.
                if do_D_r1:
                    with torch.autograd.profiler.record_function(
                            'r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()],
                                                       inputs=[real_image_tmp],
                                                       create_graph=True,
                                                       only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            if do_D_main:
                with torch.autograd.profiler.record_function('Dreal_backward'):
                    loss_Dreal.mean().mul(self.D_lambda).mul(gain).backward()

            if do_D_r1:
                with torch.autograd.profiler.record_function('Dr1_backward'):
                    # (real_logits * 0) added to solve DDP 'use all forward outputs in backward'
                    (real_logits.mean() * 0 + loss_Dr1).mean().mul(gain).backward()
