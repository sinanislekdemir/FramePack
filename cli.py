#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from PIL import Image

from inference import infer_video  # repo function


def main():
    parser = argparse.ArgumentParser("FramePack CLI (repo-native)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--first-image", required=True)
    parser.add_argument("--last-image")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    first = Image.open(args.first_image).convert("RGB")
    last = Image.open(args.last_image).convert("RGB") if args.last_image else None

    torch.manual_seed(args.seed)

    frames = infer_video(
        prompt=args.prompt,
        first_frame=first,
        last_frame=last,
        num_frames=args.num_frames,
    )

    import imageio
    imageio.mimsave(args.output, frames, fps=args.fps)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
