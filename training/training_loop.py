import torch
import time
import numpy as np
import PIL
import os
import psutil
import json
import cv2
from torch.utils.data.dataloader import DataLoader
from .sampler import InfiniteSampler
from .loss import ChunkGANLoss
from datasets import init_dataset
from utils.misc import AttributeDict, print_module_summary, format_time, nan_to_num, split_list_to_batches, weight_init
from utils import training_stats, conv2d_gradfix
from networks.encoder import EncoderNetwork
from networks.decoder import DecoderNetwork
from networks.generator import Generator
from networks.discriminator import Discriminator


def setup_snapshot_image_grid(dataset, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(5120 // dataset.image_shape[2], 7, 32)
    gh = np.clip(2880 // dataset.image_shape[1], 4, 32)

    # Show random subset of training samples.
    all_indices = list(range(len(dataset)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    datas = [dataset[i] for i in grid_indices]
    return (gw, gh), datas


def draw_bounding_boxes(images, chunks_list, color=(255, 0, 0), thickness=1):
    assert len(images) == len(chunks_list)
    output_images = []
    for image, chunks in zip(images, chunks_list):
        image = cv2.cvtColor(image.squeeze(0), cv2.COLOR_GRAY2RGB)
        for x, y, w, h in chunks:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        output_images.append(image.transpose((2, 0, 1)))
    return np.stack(output_images)


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def init_optimizer(opt_type, params, opt_args={}):
    OPTIMIZER_CLASSES = {
        'sgd': torch.optim.SGD,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam
    }

    try:
        optimizer = OPTIMIZER_CLASSES[opt_type]
    except:
        raise KeyError(f'Unknown optimizer type: {opt_type}')

    return optimizer(params, **opt_args)


def generate_sample_latents(num_samples, Enc_bg, Enc_c, device, dataset=None, seed=0):
    if dataset is not None:
        data_loader = DataLoader(dataset,
                                 batch_size=num_samples,
                                 sampler=InfiniteSampler(dataset, seed=seed),
                                 collate_fn=dataset.collect_fn)
        images, all_chunks = next(iter(data_loader))
        images = images.to(device)

        c_latents = []
        mask_images = torch.ones((images.shape[0], 1, *images.shape[2:]), device=device)
        for i, chunks in enumerate(all_chunks):
            for x, y, w, h in chunks:
                chunk = images[i:i + 1, :, y:y + h, x:x + w]
                chunk = torch.nn.functional.interpolate(chunk,
                                                        Enc_c.image_size,
                                                        mode='bilinear',
                                                        align_corners=False)
                c_latents += [Enc_c(chunk)]
                mask_images[i:i + 1, :, y:y + h, x:x + w] = 0
        assert len(c_latents) > 0
        if len(c_latents) < num_samples:
            c_latents = c_latents * (num_samples // len(c_latents) + 1)

        c_latents = torch.cat(c_latents[:num_samples])
        bg_latents = Enc_bg(images, mask_images)
    else:
        c_latents = torch.randn((num_samples, Enc_c.latent_dim), device=device)
        bg_latents = torch.randn((num_samples, Enc_bg.latent_dim), device=device)

    return bg_latents, c_latents


def generate_random_G_input(num_samples,
                            G,
                            Enc_bg,
                            Enc_c,
                            device,
                            dataset,
                            batch_size,
                            seed=0,
                            min_object_num=0,
                            max_object_num=5,
                            min_object_w=8,
                            max_object_w=24,
                            min_object_h=8,
                            max_object_h=24):
    sample_bg_z, sample_c_z = generate_sample_latents(num_samples, Enc_bg, Enc_c, device, dataset,
                                                      seed)

    sample_c_zs = []
    sample_chunks = []
    for _ in range(num_samples):
        num_objects = np.random.randint(min_object_num, max_object_num)
        chunks = []
        for _ in range(num_objects):
            w = np.random.randint(min_object_w, max_object_w)
            h = np.random.randint(min_object_h, max_object_h)
            x = np.random.randint(0, G.image_size - w)
            y = np.random.randint(0, G.image_size - h)
            chunks.append((x, y, w, h))
        chunk_zs = sample_c_z[torch.randint(len(sample_c_z), (num_objects, ))]
        sample_chunks.append(chunks)
        sample_c_zs.append(chunk_zs)

    sample_bg_z = sample_bg_z.split(batch_size)
    sample_chunks = split_list_to_batches(sample_chunks, batch_size)
    sample_c_zs = split_list_to_batches(sample_c_zs, batch_size)

    return list(zip(sample_bg_z, sample_c_zs, sample_chunks))


def generate_sample_all_inputs(num_sample,
                               G,
                               Enc_bg,
                               Enc_c,
                               dataset,
                               device,
                               batch_size,
                               sample_input_args={}):
    G_inputs = generate_random_G_input(num_sample, G, Enc_bg, Enc_c, device, dataset, batch_size,
                                       **sample_input_args)
    bg_latents, c_latents = generate_sample_latents(num_sample, Enc_bg, Enc_c, device, dataset)
    bg_latents = bg_latents.split(batch_size)
    c_latents = c_latents.split(batch_size)
    return G_inputs, bg_latents, c_latents


def training_loop(
        run_dir='.',  # Output directory.
        num_gpus=1,  # Number of GPUs participating in the training.
        gid=0,  # Index of the current process in [0, num_gpus).
        random_seed=0,  # Global random seed.
        dataset_name=None,  # Dataset object
        resume_pkl=None,  # Network pickle to resume training from.
        batch_size=4,  # Total batch size for one training iteration.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        kimg_per_tick=4,  # Progress snapshot interval.
        G_D_train_start_tick=None,  # How many ticks before training of G and D? None = start immediately.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=4,  # How often to perform regularization for D? None = disable lazy regularization.
        EncDec_c_reg_interval=None,  # How often to perform regularization for Enc_c and Dec_c? None = disable lazy regularization.
        EncDec_bg_reg_interval=None,  # How often to perform regularization for Enc_bg andDec_bg? None = disable lazy regularization.
        G_inputs_sample_interval=None,  # How often to sample G inputs? None = using random G inputs.
        dataset_args={},  # Options for dataset.
        data_loader_args={},  # Options for torch.utils.data.DataLoader.
        loss_args={},  # Options for loss function.
        network_args={},  # Options for network.
        G_opt_args={},  # Options for generator optimizer.
        D_opt_args={},  # Options for discriminator optimizer.
        EncDec_c_opt_args={},  # Options for chunk encoder/decoder optimizer.
        EncDec_bg_opt_args={},  # Options for background encoder/decoder optimizer.
        sample_input_args={},  # Options for sample images generation control.
):
    ### 1. Initialize.
    start_time = time.time()
    device = torch.device('cuda', gid)
    np.random.seed(random_seed * num_gpus + gid)
    torch.manual_seed(random_seed * num_gpus + gid)
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    conv2d_gradfix.enabled = True  # Improves training speed.

    ### 2. Load training dataset.
    dataset = init_dataset(dataset_name, dataset_args)
    data_sampler = InfiniteSampler(dataset, idx=gid, num_replicas=num_gpus, seed=random_seed)
    data_iterator = iter(
        DataLoader(dataset,
                   sampler=data_sampler,
                   batch_size=batch_size // num_gpus,
                   collate_fn=dataset.collect_fn,
                   **data_loader_args))

    ### 3. Construct networks.
    if gid == 0:
        print('Constructing networks...')
    G = Generator(dataset.image_resolution,
                  network_args.chunk_size,
                  dataset.image_channels,
                  bg_latent_dim=network_args.bg_latent_dim,
                  chunk_latent_dim=network_args.chunk_latent_dim,
                  **network_args.get('G', {})).train().requires_grad_(False).to(device)
    D = Discriminator(dataset.image_resolution, dataset.image_channels,
                      **network_args.get('D', {})).train().requires_grad_(False).to(device)
    Enc_c = EncoderNetwork(network_args.chunk_size,
                           dataset.image_channels,
                           latent_dim=network_args.chunk_latent_dim,
                           **network_args.get('Enc_c',
                                              {})).train().requires_grad_(False).to(device)
    Enc_bg = EncoderNetwork(dataset.image_resolution,
                            dataset.image_channels,
                            use_labelmap=True,
                            latent_dim=network_args.bg_latent_dim,
                            **{
                                "num_layers": 4,
                                **network_args.get('Enc_bg', {})
                            }).train().requires_grad_(False).to(device)
    Dec_c = DecoderNetwork(network_args.chunk_size,
                           dataset.image_channels,
                           latent_dim=network_args.chunk_latent_dim,
                           **network_args.get('Dec_c',
                                              {})).train().requires_grad_(False).to(device)
    Dec_bg = DecoderNetwork(dataset.image_resolution,
                            dataset.image_channels,
                            latent_dim=network_args.bg_latent_dim,
                            **{
                                "num_layers": 4,
                                **network_args.get('Dec_bg', {})
                            }).train().requires_grad_(False).to(device)
    all_modules = {
        'G': G,
        'D': D,
        'Enc_c': Enc_c,
        'Enc_bg': Enc_bg,
        'Dec_c': Dec_c,
        'Dec_bg': Dec_bg,
    }
    # Init module weights.
    for module in all_modules.values():
        module.apply(weight_init)

    # Resume from existing pickle.
    if (resume_pkl is not None) and (gid == 0):
        print(f'Resuming from "{resume_pkl}"')
        with open(resume_pkl, "rb") as f:
            resume_network = torch.load(f)
        for name, module in all_modules.items():
            module.load_state_dict(resume_network[name])

    # Print network summary tables.
    if gid == 0:
        latent_chunk = torch.empty([batch_gpu, Enc_c.latent_dim], device=device)
        latent_bg = torch.empty([batch_gpu, Enc_bg.latent_dim], device=device)
        chunk_img = print_module_summary(Dec_c, [latent_chunk])
        bg_img = print_module_summary(Dec_bg, [latent_bg])
        chunk = (bg_img.shape[2] // 2, bg_img.shape[3] // 2, chunk_img.shape[2] // 2,
                 chunk_img.shape[3] // 2)
        print_module_summary(Enc_c, [chunk_img])
        print_module_summary(
            Enc_bg,
            [bg_img, torch.ones((bg_img.shape[0], 1, *bg_img.shape[2:]), device=device)])
        image = print_module_summary(
            G, [latent_bg, latent_chunk.unsqueeze(1), batch_gpu * [[chunk]]])
        print_module_summary(D, [image])

    ### 4. Distribute across GPUs.
    if gid == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in all_modules.items():
        if (num_gpus > 1) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module,
                                                               device_ids=[device],
                                                               broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    ### 5. Setup training phases.
    if gid == 0:
        print('Setting up training phases...')
    loss = ChunkGANLoss(device, network_args.chunk_size, **ddp_modules, **loss_args)
    phases = []
    for name, modules, opt_args, reg_interval, start_tick in [
        ('EncDec_c', [Enc_c, Dec_c], EncDec_c_opt_args, EncDec_c_reg_interval, 0),
        ('EncDec_bg', [Enc_bg, Dec_bg], EncDec_bg_opt_args, EncDec_bg_reg_interval, 0),
        ('G', [G], G_opt_args, G_reg_interval, G_D_train_start_tick or 0),
        ('D', [D], D_opt_args, D_reg_interval, G_D_train_start_tick or 0),
    ]:
        opt_args = AttributeDict(opt_args)
        opt_type = opt_args.pop('type')
        if reg_interval is None:
            opts = [init_optimizer(opt_type, module.parameters(), opt_args) for module in modules]
            phases += [
                AttributeDict(name=name + '_both',
                              modules=modules,
                              opts=opts,
                              interval=1,
                              start_tick=start_tick)
            ]
        else:  # With lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_args.lr = opt_args.lr * mb_ratio
            opts = [init_optimizer(opt_type, module.parameters(), opt_args) for module in modules]
            phases += [
                AttributeDict(name=name + '_main',
                              modules=modules,
                              opts=opts,
                              interval=1,
                              start_tick=start_tick)
            ]
            phases += [
                AttributeDict(name=name + '_reg',
                              modules=modules,
                              opts=opts,
                              interval=reg_interval,
                              start_tick=start_tick)
            ]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if gid == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    ### 6. Export sample images.
    if gid == 0:
        print('Exporting sample images...')
        grid_size, datas = setup_snapshot_image_grid(dataset)
        images = np.stack([data[0] for data in datas])
        chunks_list = [data[1] for data in datas]
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), [0, 1], grid_size)
        save_image_grid(draw_bounding_boxes(images, chunks_list),
                        os.path.join(run_dir, 'reals_bbox.png'), [0, 1], grid_size)
        grid_G_inputs, grid_bg_latents, grid_c_latents = generate_sample_all_inputs(
            len(datas), G, Enc_bg, Enc_c, dataset, device, batch_gpu, sample_input_args)
        images = torch.cat(
            [G(bg_z, chunk_zs, chunks).cpu() for bg_z, chunk_zs, chunks in grid_G_inputs]).numpy()
        bg_images = torch.cat([Dec_bg(bg_latent).cpu() for bg_latent in grid_bg_latents]).numpy()
        c_images = torch.cat([
            torch.nn.functional.interpolate(Dec_c(c_latent).cpu(),
                                            images.shape[-2:],
                                            mode='bilinear',
                                            align_corners=False) for c_latent in grid_c_latents
        ]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), [-1, 1], grid_size)
        save_image_grid(bg_images, os.path.join(run_dir, 'background_init.png'), [-1, 1],
                        grid_size)
        save_image_grid(c_images, os.path.join(run_dir, 'chunk_init.png'), [-1, 1], grid_size)

    ### 7. Initialize logs.
    if gid == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if gid == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print(f'Skipping tfevents export: {err}')

    ### 8. Training loop.
    if gid == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    phase_G_inputs = None
    while True:
        ### 9. Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            data = next(data_iterator)
            # Map image pixel range from [0,1] to [-1,1]
            phase_real_img = (data[0].to(device) * 2 - 1).split(batch_gpu)
            phase_chunks = split_list_to_batches(data[1], batch_gpu)
            if G_inputs_sample_interval is None:
                phase_G_inputs = generate_random_G_input(batch_size // num_gpus,
                                                         G,
                                                         Enc_bg,
                                                         Enc_c,
                                                         device,
                                                         None,
                                                         batch_gpu,
                                                         seed=batch_idx,
                                                         **sample_input_args)
            elif batch_idx % G_inputs_sample_interval == 0:
                phase_G_inputs = generate_random_G_input(batch_size // num_gpus,
                                                         G,
                                                         Enc_bg,
                                                         Enc_c,
                                                         device,
                                                         dataset,
                                                         batch_gpu,
                                                         seed=batch_idx,
                                                         **sample_input_args)

        ### 10. Execute training phases.
        for phase in phases:
            if batch_idx % phase.interval != 0:
                continue
            if cur_tick < phase.start_tick:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            for opt in phase.opts:
                opt.zero_grad(set_to_none=True)
            for module in phase.modules:
                module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, chunks, round_G_inputs) in enumerate(
                    zip(phase_real_img, phase_chunks, phase_G_inputs)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name,
                                          real_image=real_img,
                                          chunks=chunks,
                                          G_inputs=round_G_inputs,
                                          sync=sync,
                                          gain=gain)

            # Update weights.
            for module in phase.modules:
                module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for module in phase.modules:
                    for param in module.parameters():
                        if param.grad is not None:
                            nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                for opt in phase.opts:
                    opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        ### 11. Update state.
        cur_nimg += batch_size
        batch_idx += 1
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if gid == 0:
            print(' '.join(fields))

        # 12. Save image snapshot.
        if (gid == 0) and (image_snapshot_ticks
                           is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            grid_G_inputs, grid_bg_latents, grid_c_latents = generate_sample_all_inputs(
                len(images), G, Enc_bg, Enc_c, dataset, device, batch_gpu, sample_input_args)
            images = torch.cat([
                G(bg_z, chunk_zs, chunks).cpu() for bg_z, chunk_zs, chunks in grid_G_inputs
            ]).numpy()
            chunks_list = [j for sub in [chunks for _, _, chunks in grid_G_inputs] for j in sub]
            bg_images = torch.cat([Dec_bg(bg_latent).cpu()
                                   for bg_latent in grid_bg_latents]).numpy()
            c_images = torch.cat([
                torch.nn.functional.interpolate(Dec_c(c_latent).cpu(),
                                                images.shape[-2:],
                                                mode='bilinear',
                                                align_corners=False) for c_latent in grid_c_latents
            ]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'),
                            [-1, 1], grid_size)
            save_image_grid(draw_bounding_boxes(images, chunks_list),
                            os.path.join(run_dir, f'fakes_bbox{cur_nimg//1000:06d}.png'), [-1, 1],
                            grid_size)
            save_image_grid(bg_images, os.path.join(run_dir,
                                                    f'background{cur_nimg//1000:06d}.png'),
                            [-1, 1], grid_size)
            save_image_grid(c_images, os.path.join(run_dir, f'chunk{cur_nimg//1000:06d}.png'),
                            [-1, 1], grid_size)

        # 13. Save network snapshot.
        if (network_snapshot_ticks is not None) and (done
                                                     or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict()
            for name, module in all_modules.items():
                snapshot_data[name] = module.state_dict()
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if gid == 0:
                with open(snapshot_pkl, 'wb') as f:
                    torch.save(snapshot_data, f)

        # 14. Collect statistics.
        for phase in phases:
            if cur_tick < phase.start_tick:
                continue
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # 15. Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name,
                                          value.mean,
                                          global_step=global_step,
                                          walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}',
                                          value,
                                          global_step=global_step,
                                          walltime=walltime)
            stats_tfevents.flush()

        # 16. Update maintenance state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if gid == 0:
        print()
        print('Exiting...')
