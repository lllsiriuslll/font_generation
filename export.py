# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import torch
import argparse
from model.unet import UNet

parser = argparse.ArgumentParser(description='Export generator weights from the checkpoint file')
parser.add_argument('--model_dir', dest='model_dir', required=True,
                    help='directory that saves the model checkpoints')
parser.add_argument('--inst_norm', dest='inst_norm', type=bool, default=False,
                    help='use conditional instance normalization in your model')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
# For improvement experiment
parser.add_argument('--g_norm_type', dest='g_norm_type', type=str, default="bn",
                    help='the type of Generator Norm Layer', choices=['bn', 'in', 'cbn'])
parser.add_argument('--image_size', dest='image_size', type=int, default=256, choices=[256, 512],
                    help="size of your input and output image")
args = parser.parse_args()


def main():
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    model = UNet(device, 
                 input_width=args.image_size, 
                 output_width=args.image_size, 
                 inst_norm=args.inst_norm, 
                 g_norm_type=args.g_norm_type).to(device)
    model.export_generator(save_dir=args.save_dir, model_dir=args.model_dir)

if __name__ == '__main__':
    main()
