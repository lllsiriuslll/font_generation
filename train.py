# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import torch
import argparse

from model.unet import UNet

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=str, default="0",
                    help='sequence id for the experiments you prepare to run')
parser.add_argument('--image_size', dest='image_size', type=int, default=256, choices=[256, 512],
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', dest='Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', dest='embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume', dest='resume', action='store_true', default=False, help='resume from previous training')
parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, default=0,
                    help="freeze encoder weights during training")
parser.add_argument('--fine_tune', dest='fine_tune', type=str, default=None,
                    help='specific labels id to be fine tuned')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--sample_steps', dest='sample_steps', type=int, default=10,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, default=500,
                    help='number of batches in between two checkpoints')
parser.add_argument('--flip_labels', dest='flip_labels', action='store_true', default=False,
                    help='whether flip training data labels or not, in fine tuning')
# For improvement experiment
parser.add_argument('--g_norm_type', dest='g_norm_type', type=str, default="bn",
                    help='the type of Generator Norm Layer', choices=['bn', 'in', 'cbn'])
parser.add_argument('--d_norm_type', dest='d_norm_type', type=str, default="bn",
                    help='the type of Discriminator Norm Layer', choices=['bn', 'sn'])
parser.add_argument('--gan_loss_type', dest='gan_loss_type', type=str, default="vanilla",
                    help='the type of GAN loss', choices=['vanilla', 'lsgan'])
parser.add_argument('--cycle_gan', dest='cycle_gan', action='store_true', default=False,
                    help='Cycle GAN Mode ON')
parser.add_argument('--rotate_range', type=float, default=0, help='rotate range for random rotate image')
parser.add_argument('--ignore_label', dest='ignore_label', type=str, default=None, help='ignore labels id')
args = parser.parse_args()


def main():
    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    model = UNet(device,
                 args.experiment_dir, 
                 batch_size=args.batch_size, 
                 experiment_id=args.experiment_id,
                 input_width=args.image_size, 
                 output_width=args.image_size, 
                 embedding_num=args.embedding_num,
                 embedding_dim=args.embedding_dim, 
                 L1_penalty=args.L1_penalty, 
                 Lconst_penalty=args.Lconst_penalty,
                 Ltv_penalty=args.Ltv_penalty, 
                 Lcategory_penalty=args.Lcategory_penalty, 
                 inst_norm=args.inst_norm,
                 g_norm_type=args.g_norm_type,
                 d_norm_type=args.d_norm_type,
                 gan_loss_type=args.gan_loss_type,
                 cycle_gan=args.cycle_gan,
                 rotate_range=args.rotate_range).to(device)

    fine_tune_list = None
    if args.fine_tune:
        ids = args.fine_tune.split(",")
        fine_tune_list = set([int(i) for i in ids])
    ignore_label_list = None
    if args.ignore_label:
        ids = args.ignore_label.split(",")
        ignore_label_list = set([int(i) for i in ids])
    model.train(lr=args.lr, epoch=args.epoch, resume=args.resume,
                schedule=args.schedule, freeze_encoder=args.freeze_encoder, fine_tune=fine_tune_list,
                sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps,
                flip_labels=args.flip_labels, ignore_label=ignore_label_list)


if __name__ == '__main__':
    main()
