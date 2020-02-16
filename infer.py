# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import torch
import os
import argparse
from model.unet import UNet
from model.utils import compile_frames_to_gif

parser = argparse.ArgumentParser(description='Inference for unseen data')
parser.add_argument('--model_dir', dest='model_dir', required=True,
                    help='directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--source_obj', dest='source_obj', type=str, required=True, help='the source images for inference')
parser.add_argument('--embedding_ids', default='embedding_ids', type=str, help='embeddings involved')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--interpolate', dest='interpolate', type=int, default=0,
                    help='interpolate between different embedding vectors')
parser.add_argument('--steps', dest='steps', type=int, default=10, help='interpolation steps in between vectors')
parser.add_argument('--output_gif', dest='output_gif', type=str, default=None, help='output name transition gif')
parser.add_argument('--uroboros', dest='uroboros', type=int, default=0,
                    help='Shōnen yo, you have stepped into uncharted territory')
# For improvement experiment
parser.add_argument('--g_norm_type', dest='g_norm_type', type=str, default="bn",
                    help='the type of Generator Norm Layer', choices=['bn', 'in', 'cbn'])
parser.add_argument('--rotate_range', type=float, default=0, help='rotate range for random rotate image')
parser.add_argument('--image_size', dest='image_size', type=int, default=256, choices=[256, 512],
                    help="size of your input and output image")
args = parser.parse_args()


def main():
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    model = UNet(device,
                 batch_size=args.batch_size,
                 input_width=args.image_size, 
                 output_width=args.image_size, 
                 inst_norm=args.inst_norm,
                 g_norm_type=args.g_norm_type,
                 rotate_range=args.rotate_range).to(device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print("create result save directory")

    embedding_ids = [int(i) for i in args.embedding_ids.split(",")]
    if not args.interpolate:
        if len(embedding_ids) == 1:
            embedding_ids = embedding_ids[0]
        model.infer(model_dir=args.model_dir, source_obj=args.source_obj, embedding_ids=embedding_ids,
                    save_dir=args.save_dir)
    else:
        if len(embedding_ids) < 2:
            raise Exception("no need to interpolate yourself unless you are a narcissist")
        chains = embedding_ids[:]
        if args.uroboros:
            chains.append(chains[0])
        pairs = list()
        for i in range(len(chains) - 1):
            pairs.append((chains[i], chains[i + 1]))
        for s, e in pairs:
            model.interpolate(model_dir=args.model_dir, source_obj=args.source_obj, between=[s, e],
                              save_dir=args.save_dir, steps=args.steps)
        if args.output_gif:
            gif_path = os.path.join(args.save_dir, args.output_gif)
            compile_frames_to_gif(args.save_dir, gif_path)
            print("gif saved at %s" % gif_path)


if __name__ == '__main__':
    main()
