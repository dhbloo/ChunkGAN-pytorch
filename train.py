import click
import os
import re
import json
import torch
import tempfile
from utils.misc import AttributeDict, Logger
from utils import training_stats
from datasets import init_dataset, BaseDataset
from training.training_loop import training_loop


class UserError(Exception):
    pass


def setup_training_args(
        gpus=None,  # Number of GPUs: <int>, default = 1 gpu
        snap=None,  # Snapshot interval: <int>, default = 50 ticks
        metrics=None,  # List of metric names: [], ['fid50k_full'] (default), ...
        seed=None,  # Random seed: <int>, default = 0
        dataset=None,  # Dataset name: <str>
        dataset_args=None,  # Dataset args
        resume=None,  # Resume network filename: <file>
        kimgs=None,  # Training duration: <int>
        batch=None,  # Batch size: <int>
        batch_gpu=None,  # Batch size for one GPU: <int>
        lrate=None,  # Learning rate: <float>
        chunk_size=None,  # Chunk resolution: <int>
        data_loader_args=None,  # Extra data loader args: {}
        loss_args=None,  # Extra loss args: {}
        network_args=None,  # Extra network args: {}
        G_opt_args=None,  # Extra generator optimizer args: {}.
        D_opt_args=None,  # Extra discriminator optimizer args: {}.
        EncDec_c_opt_args=None,  # Extra chunk encoder/decoder optimizer args: {}.
        EncDec_bg_opt_args=None,  # Extra background encoder/decoder optimizer args: {}.
        sample_input_args=None,  # Sample images control args: {}
        sample_interval=None,  # Sample interval: <int>
        G_D_train_start_tick=None,  # Start tick of training of G and D.: <int>
):
    args = AttributeDict()

    # -----------------------------------------------------
    # General options: gpus, snap, metrics, seed
    # -----------------------------------------------------
    args.num_gpus = gpus or 1
    if not (args.num_gpus >= 1 and args.num_gpus & (args.num_gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')

    args.image_snapshot_ticks = snap or 50
    args.network_snapshot_ticks = snap or 50
    if snap is not None and snap < 1:
        raise UserError('--snap must be at least 1')

    args.random_seed = seed or 0

    # -----------------------------------------------------
    # Dataset: dataset, data_root, data_args, resume
    # -----------------------------------------------------
    args.dataset_name = dataset
    args.dataset_args = dataset_args or {}
    args.data_loader_args = AttributeDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    if data_loader_args:
        args.data_loader_args.update(data_loader_args)
    try:
        print('Dataset info:')
        ds = init_dataset(args.dataset_name, args.dataset_args)
        assert isinstance(ds, BaseDataset)
        print(f'Training dataset:   {ds.description}')
        print(f'Image shape:        {ds.image_shape}')
        print(f'Number of images:   {len(ds)}')
        print()
        desc = f'{args.dataset_name}{ds.image_resolution}'
        res = ds.image_resolution
        del ds
    except Exception as err:
        raise UserError(f'--data: {err}')

    args.resume_pkl = resume
    if resume: desc += '-resume'

    # -----------------------------------------------------
    # Training: kimgs, batch_size, loss args, optimizer args
    # -----------------------------------------------------
    args.batch_size = batch or max(min(args.num_gpus * min(8192 // res, 64), 128),
                                   args.num_gpus)  # keep gpu memory consumption at bay
    args.batch_gpu = batch_gpu or args.batch_size // args.num_gpus
    args.total_kimg = kimgs or 25000
    args.loss_args = AttributeDict(r1_gamma=0.001 * (res**2) / args.batch_size)
    if loss_args:
        args.loss_args.update(loss_args)
    learning_rate = lrate or 0.001
    args.G_opt_args = AttributeDict(type='adam', lr=learning_rate, betas=[0, 0.99])
    if G_opt_args:
        args.G_opt_args.update(G_opt_args)
    args.D_opt_args = AttributeDict(type='adam', lr=learning_rate, betas=[0, 0.99])
    if D_opt_args:
        args.D_opt_args.update(D_opt_args)
    args.EncDec_c_opt_args = AttributeDict(type='adam', lr=learning_rate)
    if EncDec_c_opt_args:
        args.EncDec_c_opt_args.update(EncDec_c_opt_args)
    args.EncDec_bg_opt_args = AttributeDict(type='adam', lr=learning_rate)
    if EncDec_bg_opt_args:
        args.EncDec_bg_opt_args.update(EncDec_bg_opt_args)
    args.sample_input_args = AttributeDict()
    if sample_input_args:
        args.sample_input_args.update(sample_input_args)
    args.G_inputs_sample_interval = sample_interval
    args.G_D_train_start_tick = G_D_train_start_tick

    if kimgs: desc += f'-kimg{kimgs}'
    if lrate: desc += f'-lr{lrate}'

    # -----------------------------------------------------
    # Network: chunk size
    # -----------------------------------------------------
    args.network_args = AttributeDict(chunk_size=chunk_size or 32,
                                      chunk_latent_dim=512,
                                      bg_latent_dim=512)
    if network_args:
        args.network_args.update(network_args)

    if chunk_size: desc += f'-chunksize{chunk_size}'

    return args, desc


def subprocess_fn(process_idx, args, temp_dir):
    Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo',
                                                 init_method=init_method,
                                                 rank=process_idx,
                                                 world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method=init_method,
                                                 rank=process_idx,
                                                 world_size=args.num_gpus)

    # Init training stats.
    sync_device = torch.device('cuda', process_idx) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=process_idx, sync_device=sync_device)

    # Execute training loop.
    training_loop(gid=process_idx, **args)


@click.command()
# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--config', help='Path to config json file', required=True, metavar='FILE')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
def run(outdir, config, dry_run, **cmd_args):

    print("Loading config file...")
    # Load config file
    with open(config, 'r') as j:
        cfg_args = json.loads(j.read())

    user_args = {**cfg_args, **cmd_args}
    args, run_desc = setup_training_args(**user_args)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:04d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn,
                                        args=(args, temp_dir),
                                        nprocs=args.num_gpus)


if __name__ == '__main__':
    run()