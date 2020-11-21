import argparse

import numpy as np
import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm


def interpolate(sample_z, steps):
    out = np.zeros(((sample_z.shape[0] - 1) * steps, sample_z.shape[1]))
    out_index = 0
    for i in range(sample_z.shape[0] - 1):
        for index in range(steps):
            fraction = index / float(steps)
            out[out_index, :] = sample_z[i + 1, :] * fraction + sample_z[i, :] * (1 - fraction)
            out_index += 1

    return out


def generate(args, g_ema, device):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            if args.interpolate_steps > 0:
                sample_z = interpolate(np.random.randn(args.sample + 1, args.latent), args.interpolate_steps)
            else:
                sample_z = np.random.randn(args.sample, args.latent)

            sample_z = torch.from_numpy(sample_z).to(device, dtype=torch.float32)

            sample = g_ema([sample_z], truncation=args.truncation)

            utils.save_image(
                sample,
                f'sample/{str(i).zfill(6)}.png',
                nrow=args.nrow,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--nrow', type=int, default=1)
    parser.add_argument('--self_contained_checkpoint', action='store_true')
    parser.add_argument('--interpolate_steps', type=int, default=0)

    args = parser.parse_args()

    if args.self_contained_checkpoint:
        g_ema = torch.load(args.ckpt)
    else:
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        checkpoint = torch.load(args.ckpt)

        g_ema.load_state_dict(checkpoint['g_ema'])

    generate(args, g_ema, args.device)
