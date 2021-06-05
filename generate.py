import torch
import numpy as np
import os
import click
import json
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from training.sampler import InfiniteSampler
from datasets import init_dataset
from utils.misc import AttributeDict, numpy_to_image, weight_init
from networks.encoder import EncoderNetwork
from networks.decoder import DecoderNetwork
from networks.generator import Generator
from training.training_loop import generate_sample_G_inputs, draw_bounding_boxes, generate_sample_latents

#----------------------------------------------------------------------------


def setup_networks(
        dataset,
        device,
        chunk_size,
        network_pkl=None,
        network_args={},  # Options for network.
):
    torch.backends.cudnn.benchmark = True  # Improves training speed.

    init_args = AttributeDict(chunk_size=chunk_size, chunk_latent_dim=512, bg_latent_dim=512)
    init_args.update(network_args)
    network_args = init_args

    G = Generator(dataset.image_resolution,
                  network_args.chunk_size,
                  dataset.image_channels,
                  bg_latent_dim=network_args.bg_latent_dim,
                  chunk_latent_dim=network_args.chunk_latent_dim,
                  **network_args.get('G', {})).train().requires_grad_(False).to(device)
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
    # Dec_c = DecoderNetwork(network_args.chunk_size,
    #                        dataset.image_channels,
    #                        latent_dim=network_args.chunk_latent_dim,
    #                        **network_args.get('Dec_c', {})).eval().requires_grad_(False).to(device)
    # Dec_bg = DecoderNetwork(dataset.image_resolution,
    #                         dataset.image_channels,
    #                         latent_dim=network_args.bg_latent_dim,
    #                         **{
    #                             "num_layers": 4,
    #                             **network_args.get('Dec_bg', {})
    #                         }).eval().requires_grad_(False).to(device)
    all_modules = {
        'G': G,
        'Enc_c': Enc_c,
        'Enc_bg': Enc_bg,
        #'Dec_c': Dec_c,
        #'Dec_bg': Dec_bg,
    }
    # Init module weights.
    for module in all_modules.values():
        module.apply(weight_init)

    # Load from existing pickle.
    if network_pkl is not None:
        print(f'Loading networks from {network_pkl}...')
        with open(network_pkl, "rb") as f:
            snapshot_data = torch.load(f)
        for name, module in all_modules.items():
            module.load_state_dict(snapshot_data[name])

            #module.eval()
            def set_BatchNorm2d_momentum(m):
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.momentum = 0

            module.apply(set_BatchNorm2d_momentum)

    return all_modules


def translate_chunks(chunks, tx, ty, image_size=128):
    translated_chunks = []
    for x, y, w, h in chunks:
        x = max(0, min(image_size - 1 - w, x + tx))
        y = max(0, min(image_size - 1 - h, y + ty))
        translated_chunks += [(x, y, w, h)]
    return translated_chunks


def gen_sample_chunks(c_latents, chunk_latents, chunks, num_objs, image_size=128, seed=0):
    np.random.seed(seed)
    chunks = []
    for _ in range(num_objs):
        w = np.random.randint(8, 24)
        h = np.random.randint(8, 24)
        x = np.random.randint(0, image_size - w)
        y = np.random.randint(0, image_size - h)
        chunks.append((x, y, w, h))
        
    chunk_zs = c_latents[torch.randint(len(c_latents), (num_objs, ))]

    return chunk_zs, chunks



@click.command()
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--config', help='Path to config json file', required=True, metavar='FILE')
@click.option('--network',
              'network_pkl',
              help='Path to trained network file',
              required=True,
              metavar='FILE')
@click.option('--seed', type=int, default=0)
def run(outdir, config, network_pkl, seed, batch_size=64):

    device = torch.device('cuda', 0)

    print("Loading config file...")
    # Load config file
    with open(config, 'r') as j:
        args = AttributeDict(json.loads(j.read()))

    dataset = init_dataset(args.dataset, args.dataset_args or {})

    all_modules = setup_networks(dataset,
                                 device,
                                 chunk_size=args.get("chunk_size", 32),
                                 network_pkl=network_pkl,
                                 network_args=args.get("network_args", {}))
    G = all_modules['G']
    Enc_c = all_modules['Enc_c']
    Enc_bg = all_modules['Enc_bg']

    G_inputs = generate_sample_G_inputs(batch_size,
                                        Enc_bg,
                                        Enc_c,
                                        device,
                                        dataset,
                                        batch_size=batch_size,
                                        seed=seed)
    _, c_latents = generate_sample_latents(batch_size, Enc_bg, Enc_c, device, dataset, seed=seed)
    image_idx = 2

    # for t in np.linspace(0, 1, 10):
    #     bg_latent, chunk_latents, chunks = G_inputs[0]
    #     tx = int(t * 60)
    #     ty = int(t * 2)
    #     chunks[image_idx] = translate_chunks(chunks[image_idx], tx, ty)

    #     fake_image = G(bg_latent, chunk_latents, chunks).cpu().numpy()
    #     fake_image_bbox = draw_bounding_boxes(fake_image, chunks)
    #     fake_image = np.repeat(np.rint((fake_image + 1) * 127.5).clip(0, 255), 3, axis=1)
    #     fake_image_bbox = np.rint((fake_image_bbox + 1) * 127.5).clip(0, 255)

    #     #numpy_to_image(fake_image[image_idx]).save(f'{outdir}/fake{image_idx}.png')
    #     numpy_to_image(fake_image_bbox[image_idx]).save(f'{outdir}/fake_bbox{image_idx}-t={t:.2f}.png')

    chunk_latents_list = c_latents[[0,16,39,23]]
    chunks_list = [
        (20, 20, 24, 16),
        (50, 40, 16, 16),
        (30, 90, 28, 16),
        (100, 46, 20, 16),
    ]

    for num_objs in range(0, 5):
        bg_latent, chunk_latents, chunks = G_inputs[0]
        #chunk_latents[image_idx], chunks[image_idx] = gen_sample_chunks(c_latents,
        #    chunk_latents[image_idx], chunks[image_idx], num_objs, seed=9)
        chunk_latents[image_idx] = chunk_latents_list[:num_objs]
        chunks[image_idx] = chunks_list[:num_objs]

        fake_image = G(bg_latent, chunk_latents, chunks).cpu().numpy()
        fake_image_bbox = draw_bounding_boxes(fake_image, chunks)
        fake_image = np.repeat(np.rint((fake_image + 1) * 127.5).clip(0, 255), 3, axis=1)
        fake_image_bbox = np.rint((fake_image_bbox + 1) * 127.5).clip(0, 255)

        #numpy_to_image(fake_image[image_idx]).save(f'{outdir}/fake{image_idx}.png')
        numpy_to_image(
            fake_image_bbox[image_idx]).save(f'{outdir}/fake_bbox{image_idx}-objs={num_objs}.png')


if __name__ == '__main__':
    run()