# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import itertools
import numpy as np
import scipy.misc as misc
import os
import time
from collections import namedtuple
from .ops import Conv2d, Deconv2d, Lrelu, FC, BatchNorm, Embedding, ConditionalInstanceNorm
from .ops import InstanceNorm, ConditionalBatchNorm, SpectralNorm
from .dataset import TrainDataProvider, InjectDataProvider, NeverEndingLoopingProvider
from .utils import scale_back, merge, save_concat_images

class Encoder(nn.Module):
    def __init__(self, input_dim, generator_dim, embedding_num, norm_type):
        super(Encoder, self).__init__()
        self.conv2d = Conv2d(input_dim, generator_dim)
        self.encode_layer1 = EncodeLayer(generator_dim, generator_dim*2, embedding_num, norm_type)
        self.encode_layer2 = EncodeLayer(generator_dim*2, generator_dim*4, embedding_num, norm_type)
        self.encode_layer3 = EncodeLayer(generator_dim*4, generator_dim*8, embedding_num, norm_type)
        self.encode_layer4 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)
        self.encode_layer5 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)
        self.encode_layer6 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)
        self.encode_layer7 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)

    def forward(self, images, one_hot_ids):
        encode_layers = dict()

        e1 = self.conv2d(images)
        e2 = self.encode_layer1(e1, one_hot_ids)
        e3 = self.encode_layer2(e2, one_hot_ids)
        e4 = self.encode_layer3(e3, one_hot_ids)
        e5 = self.encode_layer4(e4, one_hot_ids)
        e6 = self.encode_layer5(e5, one_hot_ids)
        e7 = self.encode_layer6(e6, one_hot_ids)
        e8 = self.encode_layer7(e7, one_hot_ids)

        encode_layers["e1"] = e1
        encode_layers["e2"] = e2
        encode_layers["e3"] = e3
        encode_layers["e4"] = e4
        encode_layers["e5"] = e5
        encode_layers["e6"] = e6
        encode_layers["e7"] = e7
        encode_layers["e8"] = e8

        return e8, encode_layers

class Encoder512(nn.Module):
    def __init__(self, input_dim, generator_dim, embedding_num, norm_type):
        super(Encoder512, self).__init__()
        self.conv2d = Conv2d(input_dim, generator_dim)
        self.encode_layer1 = EncodeLayer(generator_dim, generator_dim*2, embedding_num, norm_type)
        self.encode_layer2 = EncodeLayer(generator_dim*2, generator_dim*4, embedding_num, norm_type)
        self.encode_layer3 = EncodeLayer(generator_dim*4, generator_dim*8, embedding_num, norm_type)
        self.encode_layer4 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)
        self.encode_layer5 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)
        self.encode_layer6 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)
        self.encode_layer7 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)
        self.encode_layer8 = EncodeLayer(generator_dim*8, generator_dim*8, embedding_num, norm_type)

    def forward(self, images, one_hot_ids):
        encode_layers = dict()

        e1 = self.conv2d(images)
        e2 = self.encode_layer1(e1, one_hot_ids)
        e3 = self.encode_layer2(e2, one_hot_ids)
        e4 = self.encode_layer3(e3, one_hot_ids)
        e5 = self.encode_layer4(e4, one_hot_ids)
        e6 = self.encode_layer5(e5, one_hot_ids)
        e7 = self.encode_layer6(e6, one_hot_ids)
        e8 = self.encode_layer7(e7, one_hot_ids)
        e9 = self.encode_layer8(e8, one_hot_ids)

        encode_layers["e1"] = e1
        encode_layers["e2"] = e2
        encode_layers["e3"] = e3
        encode_layers["e4"] = e4
        encode_layers["e5"] = e5
        encode_layers["e6"] = e6
        encode_layers["e7"] = e7
        encode_layers["e8"] = e8
        encode_layers["e9"] = e9

        return e9, encode_layers

class EncodeLayer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_num, norm_type):
        super(EncodeLayer, self).__init__()
        self.lrelu = Lrelu()
        self.conv2d = Conv2d(input_dim, output_dim)
        if norm_type == "bn":
            self.norm = BatchNorm(output_dim)
        elif norm_type == "in":
            self.norm = InstanceNorm(output_dim)
        elif norm_type == "cbn":
            self.norm = ConditionalBatchNorm(output_dim, embedding_num)
        self.norm_type = norm_type

    def forward(self, x, one_hot_ids):
        act = self.lrelu(x)
        conv = self.conv2d(act)
        if self.norm_type == "cbn":
            enc = self.norm(conv, one_hot_ids)
        else:
            enc = self.norm(conv)
        return enc

class Decoder(nn.Module):
    def __init__(self, input_dim, generator_dim, output_dim, inst_norm, embedding_num, norm_type):
        super(Decoder, self).__init__()
        self.decode_layer1 = DecodeLayer(input_dim, generator_dim*8, inst_norm, embedding_num, dropout=True, norm_type=norm_type)
        self.decode_layer2 = DecodeLayer(generator_dim*8*2, generator_dim*8, inst_norm, embedding_num, dropout=True, norm_type=norm_type)
        self.decode_layer3 = DecodeLayer(generator_dim*8*2, generator_dim*8, inst_norm, embedding_num, dropout=True, norm_type=norm_type)
        self.decode_layer4 = DecodeLayer(generator_dim*8*2, generator_dim*8, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer5 = DecodeLayer(generator_dim*8*2, generator_dim*4, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer6 = DecodeLayer(generator_dim*4*2, generator_dim*2, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer7 = DecodeLayer(generator_dim*2*2, generator_dim, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer8 = DecodeLayer(generator_dim*2, output_dim, inst_norm, embedding_num, do_concat=False, norm_type=norm_type)

    def forward(self, encoded, encoding_layers, ids, one_hot_ids):
        d1 = self.decode_layer1(encoded, False, encoding_layers["e7"], ids, one_hot_ids)
        d2 = self.decode_layer2(d1, False, encoding_layers["e6"], ids, one_hot_ids)
        d3 = self.decode_layer3(d2, False, encoding_layers["e5"], ids, one_hot_ids)
        d4 = self.decode_layer4(d3, False, encoding_layers["e4"], ids, one_hot_ids)
        d5 = self.decode_layer5(d4, False, encoding_layers["e3"], ids, one_hot_ids)
        d6 = self.decode_layer6(d5, False, encoding_layers["e2"], ids, one_hot_ids)
        d7 = self.decode_layer7(d6, False, encoding_layers["e1"], ids, one_hot_ids)
        d8 = self.decode_layer8(d7, True, None, ids, one_hot_ids)

        output = torch.tanh(d8)
        return output

class Decoder512(nn.Module):
    def __init__(self, input_dim, generator_dim, output_dim, inst_norm, embedding_num, norm_type):
        super(Decoder512, self).__init__()
        self.decode_layer1 = DecodeLayer(input_dim, generator_dim*8, inst_norm, embedding_num, dropout=True, norm_type=norm_type)
        self.decode_layer2 = DecodeLayer(generator_dim*8*2, generator_dim*8, inst_norm, embedding_num, dropout=True, norm_type=norm_type)
        self.decode_layer3 = DecodeLayer(generator_dim*8*2, generator_dim*8, inst_norm, embedding_num, dropout=True, norm_type=norm_type)
        self.decode_layer4 = DecodeLayer(generator_dim*8*2, generator_dim*8, inst_norm, embedding_num, dropout=True, norm_type=norm_type)
        self.decode_layer5 = DecodeLayer(generator_dim*8*2, generator_dim*8, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer6 = DecodeLayer(generator_dim*8*2, generator_dim*4, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer7 = DecodeLayer(generator_dim*4*2, generator_dim*2, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer8 = DecodeLayer(generator_dim*2*2, generator_dim, inst_norm, embedding_num, norm_type=norm_type)
        self.decode_layer9 = DecodeLayer(generator_dim*2, output_dim, inst_norm, embedding_num, do_concat=False, norm_type=norm_type)

    def forward(self, encoded, encoding_layers, ids, one_hot_ids):
        d1 = self.decode_layer1(encoded, False, encoding_layers["e8"], ids, one_hot_ids)
        d2 = self.decode_layer2(d1, False, encoding_layers["e7"], ids, one_hot_ids)
        d3 = self.decode_layer3(d2, False, encoding_layers["e6"], ids, one_hot_ids)
        d4 = self.decode_layer4(d3, False, encoding_layers["e5"], ids, one_hot_ids)
        d5 = self.decode_layer5(d4, False, encoding_layers["e4"], ids, one_hot_ids)
        d6 = self.decode_layer6(d5, False, encoding_layers["e3"], ids, one_hot_ids)
        d7 = self.decode_layer7(d6, False, encoding_layers["e2"], ids, one_hot_ids)
        d8 = self.decode_layer8(d7, False, encoding_layers["e1"], ids, one_hot_ids)
        d9 = self.decode_layer9(d8, True, None, ids, one_hot_ids)

        output = torch.tanh(d9)
        return output

class DecodeLayer(nn.Module):
    def __init__(self, input_dim, output_dim, inst_norm, embedding_num, dropout=False, do_concat=True, norm_type="bn"):
        super(DecodeLayer, self).__init__()
        self.deconv2d = Deconv2d(input_dim, output_dim)
        self.conditional_instance_norm = ConditionalInstanceNorm(embedding_num, output_dim)
        if norm_type == "bn":
            self.norm = BatchNorm(output_dim)
        elif norm_type == "in":
            self.norm = InstanceNorm(output_dim)
        elif norm_type == "cbn":
            self.norm = ConditionalBatchNorm(output_dim, embedding_num)
        self.dropout = dropout
        self.do_concat = do_concat
        self.inst_norm = inst_norm
        self.instance_norm = InstanceNorm(output_dim)
        self.norm_type = norm_type

    def forward(self, x, last_layer, enc_layer, ids, one_hot_ids):
        dec = self.deconv2d(F.relu(x))
        if not last_layer:
            # IMPORTANT: normalization for last layer
            # Very important, otherwise GAN is unstable
            # Trying conditional instance normalization to
            # overcome the fact that batch normalization offers
            # different train/test statistics
            if self.inst_norm:
                dec = self.conditional_instance_norm(dec, ids)
            else:
                if self.norm_type == "cbn":
                    dec = self.norm(dec, one_hot_ids)
                else:
                    dec = self.norm(dec)
        if self.dropout:
            dec = F.dropout(dec, 0.5)
        if self.do_concat:
            dec = torch.cat((dec, enc_layer), 1)
        return dec

class Generator(nn.Module):
    def __init__(self, enc_input_dim, dec_input_dim, dec_output_dim, generator_dim, 
                 batch_size, embedding_num, embedding_dim, inst_norm, norm_type, input_size):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        if batch_size == 1:
            norm_type = "in"

        self.embeddings = Embedding(embedding_num, embedding_dim)
        if input_size == 256:
            self.encoder = Encoder(enc_input_dim, generator_dim, embedding_num, norm_type)
            self.decoder = Decoder(dec_input_dim, generator_dim, dec_output_dim, inst_norm, embedding_num, norm_type)
        elif input_size == 512:
            self.encoder = Encoder512(enc_input_dim, generator_dim, embedding_num, norm_type)
            self.decoder = Decoder512(dec_input_dim, generator_dim, dec_output_dim, inst_norm, embedding_num, norm_type)

    def forward(self, images, embedding_ids, one_hot_ids):
        z, enc_layers = self.encoder(images, one_hot_ids)
        lookup_tensor = embedding_ids
        local_embeddings = self.embeddings(lookup_tensor)
        local_embeddings = local_embeddings.view(self.batch_size, self.embedding_dim, 1, 1)
        embedded = torch.cat([z, local_embeddings], 1)
        output = self.decoder(embedded, enc_layers, embedding_ids, one_hot_ids)

        return output, z

class Discriminator(nn.Module):
    def __init__(self, input_dim, discriminator_dim, embedding_num, input_width, norm_type):
        super(Discriminator, self).__init__()
        self.lrelu = Lrelu()

        self.conv0 = Conv2d(input_dim, discriminator_dim)
        self.conv1 = Conv2d(discriminator_dim, discriminator_dim*2)
        self.conv2 = Conv2d(discriminator_dim*2, discriminator_dim*4)
        self.conv3 = Conv2d(discriminator_dim*4, discriminator_dim*8, stride=1)
        if norm_type == "bn" :
            self.norm1 = BatchNorm(discriminator_dim*2)
            self.norm2 = BatchNorm(discriminator_dim*4)
            self.norm3 = BatchNorm(discriminator_dim*8)
        elif norm_type == "sn":
            self.sn_conv1 = SpectralNorm(self.conv1.conv2d)
            self.sn_conv2 = SpectralNorm(self.conv2.conv2d)
            self.sn_conv3 = SpectralNorm(self.conv3.conv2d)

        num_features = int(discriminator_dim*8*(input_width/8)*(input_width/8))

        self.fc1 = FC(num_features, 1)
        self.fc2 = FC(num_features, embedding_num)

        self.norm_type = norm_type

    def forward(self, image, y=None):
        h0 = self.lrelu(self.conv0(image))
        if self.norm_type == "bn":
            h1 = self.lrelu(self.norm1(self.conv1(h0)))
            h2 = self.lrelu(self.norm2(self.conv2(h1)))
            h3 = self.lrelu(self.norm3(self.conv3(h2)))
        elif self.norm_type == "sn":
            h1 = self.lrelu(self.sn_conv1(self.conv1.same_padding(h0)))
            h2 = self.lrelu(self.sn_conv2(self.conv2.same_padding(h1)))
            h3 = self.lrelu(self.sn_conv3(self.conv3.same_padding(h2)))

        # real or fake binary loss
        fc1 = self.fc1(h3.view(h3.shape[0], -1))
        # category loss
        fc2 = self.fc2(h3.view(h3.shape[0], -1))

        return torch.sigmoid(fc1), fc1, fc2

class UNet(nn.Module):
    def __init__(self,  device, experiment_dir=None, experiment_id=0, batch_size=16, input_width=256, output_width=256,
                 generator_dim=64, discriminator_dim=64, L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0,
                 Lcategory_penalty=1.0, embedding_num=40, embedding_dim=128, input_filters=3, output_filters=3,
                 inst_norm=False, g_norm_type="bn", d_norm_type="bn", gan_loss_type="vanilla", cycle_gan=False,
                 rotate_range=0):
        super(UNet, self).__init__()
        self.device = device
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.batch_size = batch_size
        self.input_width = input_width
        self.output_width = output_width
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.embedding_num = embedding_num
        self.embedding_dim = embedding_dim
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.inst_norm = inst_norm

        self.g_norm_type = g_norm_type
        self.d_norm_type = d_norm_type    
        self.gan_loss_type = gan_loss_type
        self.cycle_gan = cycle_gan
        self.rotate_range = rotate_range

        dec_input_dim = generator_dim*8 + embedding_dim
        
        self.generator = Generator(input_filters, dec_input_dim, output_filters, generator_dim, 
                                   batch_size, embedding_num, embedding_dim, inst_norm, g_norm_type, input_width)
        self.discriminator = Discriminator(input_filters*2, discriminator_dim, embedding_num, input_width, d_norm_type)

        self.bcewl_loss = nn.BCEWithLogitsLoss()        

        if self.gan_loss_type == "lsgan":
            self.gan_loss = nn.MSELoss()
        else:
            self.gan_loss = nn.BCEWithLogitsLoss()

        if self.cycle_gan:
            self.c_generator = Generator(input_filters, dec_input_dim, output_filters, generator_dim, 
                                         batch_size, embedding_num, embedding_dim, inst_norm, g_norm_type, input_width)
            self.c_discriminator = Discriminator(input_filters*2, discriminator_dim, embedding_num, input_width, d_norm_type)

        # experiment_dir is needed for training
        if experiment_dir:
            self.data_dir = os.path.join(self.experiment_dir, "data")
            self.checkpoint_dir = os.path.join(self.experiment_dir, "ckpt")
            self.sample_dir = os.path.join(self.experiment_dir, "sample")
            self.log_dir = os.path.join(self.experiment_dir, "logs")

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print("create checkpoint directory")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print("create log directory")
            if not os.path.exists(self.sample_dir):
                os.makedirs(self.sample_dir)
                print("create sample directory")

    def one_hot(self, indices, depth):
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).to(self.device)
        index = indices.view(indices.size()+torch.Size([1]))
        encoded_indicies = encoded_indicies.scatter_(1,index,1)

        return encoded_indicies

    def forward(self, real_data, embedding_ids, no_target_data, no_target_ids, no_target_source=False):
        # target images
        self.real_B = real_data[:, :self.input_filters, :, :]
        # source images
        self.real_A = real_data[:, self.input_filters:self.input_filters + self.output_filters, :, :]

        self.true_labels = torch.reshape(self.one_hot(indices=embedding_ids, depth=self.embedding_num),
                                         shape=(self.batch_size, self.embedding_num))

        self.fake_B, self.encoded_real_A = self.generator(self.real_A, embedding_ids, self.true_labels)
        self.real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        self.encoded_fake_B = self.generator.encoder(self.fake_B, self.true_labels)[0]

        if self.cycle_gan:
            self.rec_real_A, self.c_encoded_fake_B = self.c_generator(self.fake_B, embedding_ids, self.true_labels)
            self.fake_A, self.encoded_real_B = self.c_generator(self.real_B, embedding_ids, self.true_labels)
            self.rec_real_B, self.encoded_fake_A = self.generator(self.fake_A, embedding_ids, self.true_labels)
            self.real_BA = torch.cat((self.real_B, self.real_A), 1)
            self.fake_BA = torch.cat((self.real_B, self.fake_A), 1)
            self.c_encoded_fake_A = self.c_generator.encoder(self.fake_A, self.true_labels)[0]

        self.no_target_source = no_target_source
        if self.no_target_source:
            # no_target source are examples that don't have the corresponding target images
            # however, except L1 loss, we can compute category loss, binary loss and constant losses with those examples
            # it is useful when discriminator get saturated and d_loss drops to near zero
            # those data could be used as additional source of losses to break the saturation
            no_target_A = no_target_data[:, self.input_filters:self.input_filters + self.output_filters, :, :]
            self.no_target_labels = torch.reshape(self.one_hot(indices=no_target_ids, depth=self.embedding_num),
                                                  shape=(self.batch_size, self.embedding_num))
            self.no_target_B, self.encoded_no_target_A = self.generator(no_target_A, no_target_ids, self.no_target_labels)
            self.no_target_AB = torch.cat((no_target_A, self.no_target_B), 1)

            self.encoded_no_target_B = self.generator.encoder(self.no_target_B, self.no_target_labels)[0]

            if self.cycle_gan:
                self.c_no_target_B, self.encoded_no_target_A = self.c_generator(no_target_A, no_target_ids, self.no_target_labels)
                self.c_no_target_AB = torch.cat((no_target_A, self.c_no_target_B), 1)

    def d_backward(self, is_training=True):
        self.d_summary = {}
        real_D, real_D_logits, real_category_logits = self.discriminator(self.real_AB, self.true_labels)
        fake_D, fake_D_logits, fake_category_logits = self.discriminator(self.fake_AB.detach(), self.true_labels)

        # category loss
        real_category_loss = torch.mean(self.gan_loss(real_category_logits, self.true_labels))
        fake_category_loss = torch.mean(self.gan_loss(fake_category_logits, self.true_labels))
        category_loss = self.Lcategory_penalty * (real_category_loss + fake_category_loss)

        self.d_summary["category_loss"] = category_loss

        # binary real/fake loss
        d_loss_real = torch.mean(self.gan_loss(real_D_logits, torch.ones_like(real_D)))
        d_loss_fake = torch.mean(self.gan_loss(fake_D_logits, torch.zeros_like(fake_D)))

        self.d_summary["d_loss_real"] = d_loss_real
        self.d_summary["d_loss_fake"] = d_loss_fake

        d_loss = d_loss_real + d_loss_fake + category_loss / 2.0

        if self.no_target_source:
            self.no_target_D, self.no_target_D_logits, no_target_category_logits = self.discriminator(self.no_target_AB.detach(), self.no_target_labels)
            self.no_target_category_loss = torch.mean(self.gan_loss(no_target_category_logits, self.no_target_labels)) * self.Lcategory_penalty
            d_loss_no_target = torch.mean(self.gan_loss(self.no_target_D_logits, torch.zeros_like(self.no_target_D)))
            d_loss = d_loss_real + d_loss_fake + d_loss_no_target + (category_loss + self.no_target_category_loss) / 3.0


        if self.cycle_gan:
            c_real_D, c_real_D_logits, c_real_category_logits = self.c_discriminator(self.real_BA, self.true_labels)
            c_fake_D, c_fake_D_logits, c_fake_category_logits = self.c_discriminator(self.fake_BA.detach(), self.true_labels)

            # cycle category loss
            c_real_category_loss = torch.mean(self.gan_loss(c_real_category_logits, self.true_labels))
            c_fake_category_loss = torch.mean(self.gan_loss(c_fake_category_logits, self.true_labels))
            c_category_loss = self.Lcategory_penalty * (c_real_category_loss + c_fake_category_loss)

            # cycle binary real/fake loss
            c_d_loss_real = torch.mean(self.gan_loss(c_real_D_logits, torch.ones_like(c_real_D)))
            c_d_loss_fake = torch.mean(self.gan_loss(c_fake_D_logits, torch.zeros_like(c_fake_D)))

            c_d_loss = c_d_loss_real + c_d_loss_fake + c_category_loss / 2.0

            if self.no_target_source:
                self.c_no_target_D, self.c_no_target_D_logits, c_no_target_category_logits = self.c_discriminator(self.c_no_target_AB, self.no_target_labels)
                self.c_no_target_category_loss = torch.mean(self.gan_loss(c_no_target_category_logits, self.no_target_labels)) * self.Lcategory_penalty
                c_d_loss_no_target = torch.mean(self.gan_loss(self.c_no_target_D_logits, torch.zeros_like(self.c_no_target_D)))
                c_d_loss = c_d_loss_real + c_d_loss_fake + c_d_loss_no_target + (c_category_loss + self.c_no_target_category_loss) / 3.0

            d_loss += c_d_loss
            self.d_summary["c_d_loss"] = c_d_loss

        if is_training:
            d_loss.backward(retain_graph=True)

        self.d_summary["d_loss"] = d_loss

    def g_backward(self, is_training=True, retain_graph=False):
        self.g_summary = {}
        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        const_loss = torch.mean(torch.pow(self.encoded_real_A - self.encoded_fake_B, 2)) * self.Lconst_penalty
        self.g_summary["const_loss"] = const_loss

        fake_D, fake_D_logits, fake_category_logits = self.discriminator(self.fake_AB, self.true_labels)
        fake_category_loss = torch.mean(self.gan_loss(fake_category_logits, self.true_labels))
        self.g_summary["fake_category_loss"] = fake_category_loss

        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * torch.mean(torch.abs(self.fake_B - self.real_B))
        self.g_summary["l1_loss"] = l1_loss

        # maximize the chance generator fool the discriminator
        cheat_loss = torch.mean(self.gan_loss(fake_D_logits, torch.ones_like(fake_D)))

        g_loss = cheat_loss + l1_loss + self.Lcategory_penalty * fake_category_loss + const_loss

        if self.no_target_source:
            no_target_const_loss = torch.mean(torch.pow(self.encoded_no_target_A - self.encoded_no_target_B, 2)) * self.Lconst_penalty
            cheat_loss += torch.mean(self.gan_loss(self.no_target_D_logits, torch.ones_like(self.no_target_D)))
            g_loss = cheat_loss / 2.0 + l1_loss + \
                    (self.Lcategory_penalty * fake_category_loss + self.no_target_category_loss) / 2.0 + \
                    (const_loss + no_target_const_loss) / 2.0

        self.g_summary["cheat_loss"] = cheat_loss

        # total variation loss
        if self.Ltv_penalty > 0:
            width = self.output_width
            tv_loss = (((torch.pow(self.fake_B[:, :, 1:, :] - self.fake_B[:, :, :width - 1, :], 2).sum() / 2) / width)
                      +((torch.pow(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :width - 1], 2).sum() / 2) / width)) * self.Ltv_penalty
            g_loss += tv_loss
            self.g_summary["tv_loss"] = tv_loss

        if self.cycle_gan:
            # cycle loss between real and reconstructed images
            cycle_loss_A = self.L1_penalty/2 * torch.mean(torch.abs(self.rec_real_A - self.real_A))
            cycle_loss_B = self.L1_penalty/2 * torch.mean(torch.abs(self.rec_real_B - self.real_B))
            cycle_loss = (cycle_loss_A + cycle_loss_B)
            self.g_summary["cycle_loss"] = cycle_loss

            # cycle constant loss
            c_const_loss = torch.mean(torch.pow(self.encoded_real_B - self.c_encoded_fake_A, 2)) * self.Lconst_penalty
            self.g_summary["c_const_loss"] = c_const_loss

            c_fake_D, c_fake_D_logits, c_fake_category_logits = self.c_discriminator(self.fake_BA, self.true_labels)
            c_fake_category_loss = torch.mean(self.gan_loss(c_fake_category_logits, self.true_labels))
            self.g_summary["c_fake_category_loss"] = c_fake_category_loss

            # cycle L1 loss between real and generated images
            c_l1_loss = self.L1_penalty * torch.mean(torch.abs(self.fake_A - self.real_A))
            self.g_summary["c_l1_loss"] = c_l1_loss

            # cycle cheat loss
            c_cheat_loss = torch.mean(self.gan_loss(c_fake_D_logits, torch.ones_like(c_fake_D)))

            c_g_loss = cycle_loss + c_cheat_loss + c_l1_loss + self.Lcategory_penalty * c_fake_category_loss + c_const_loss
    
            if self.no_target_source:
                c_no_target_const_loss = torch.mean(torch.pow(self.encoded_no_target_A - self.encoded_no_target_B, 2)) * self.Lconst_penalty
                c_cheat_loss += torch.mean(self.gan_loss(self.c_no_target_D_logits, torch.ones_like(self.c_no_target_D)))
                c_g_loss = c_cheat_loss / 2.0 + c_l1_loss + \
                          (self.Lcategory_penalty * c_fake_category_loss + self.c_no_target_category_loss) / 2.0 + \
                          (const_loss + c_no_target_const_loss) / 2.0

            self.g_summary["c_cheat_loss"] = c_cheat_loss

            # cycle total variation loss
            if self.Ltv_penalty > 0:
                width = self.output_width
                c_tv_loss = (((torch.pow(self.fake_A[:, :, 1:, :] - self.fake_A[:, :, :width - 1, :], 2).sum() / 2) / width)
                            +((torch.pow(self.fake_A[:, :, :, 1:] - self.fake_A[:, :, :, :width - 1], 2).sum() / 2) / width)) * self.Ltv_penalty
                g_loss += c_tv_loss
                self.g_summary["c_tv_loss"] = c_tv_loss

            g_loss += c_g_loss
            self.g_summary["c_g_loss"] = c_g_loss

        if is_training:
            g_loss.backward(retain_graph=retain_graph)

        self.g_summary["g_loss"] = g_loss

    def set_requires_grad(self, model, requires_grad=False):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def freeze_encoder(self):
        # freeze encoder weights
        print("freeze encoder weights")
        self.generator.encoder.eval()
        self.set_requires_grad(self.generator.encoder, False)

    def get_save_dir(self, root_dir):
        save_id= "%s_batch_%d" % (self.experiment_id, self.batch_size)
        save_dir = os.path.join(root_dir, save_id)
        return save_dir

    def checkpoint(self):
        model_dir = self.get_save_dir(self.checkpoint_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model = {'generator': self.generator.state_dict(),
                      'discriminator': self.discriminator.state_dict()}
        if self.cycle_gan:
            save_model['c_generator'] = self.c_generator.state_dict()
            save_model['c_discriminator'] = self.c_discriminator.state_dict()

        save_optimizer = {'generator': self.g_optimizer.state_dict(),
                          'discriminator': self.d_optimizer.state_dict()}

        torch.save(save_model, os.path.join(model_dir, 'model.pth'))
        torch.save(save_optimizer, os.path.join(model_dir, 'optimizer.pth'))

    def restore_model(self, model_dir, is_training=True):
        checkpoint_path = os.path.join(model_dir, 'model.pth')
        optimizer_path = os.path.join(model_dir, 'optimizer.pth')

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path)
            self.generator.load_state_dict(ckpt['generator'])
            if is_training:
                self.discriminator.load_state_dict(ckpt['discriminator'])
                if self.cycle_gan:
                    self.c_generator.load_state_dict(ckpt['c_generator'])
                    self.c_discriminator.load_state_dict(ckpt['c_discriminator'])

                if os.path.exists(optimizer_path):
                    optim_state = torch.load(optimizer_path)
                    self.g_optimizer.load_state_dict(optim_state['generator'])
                    self.d_optimizer.load_state_dict(optim_state['discriminator'])
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)
            exit()

    def validate_model(self, val_iter, epoch, step):
        # set validation mode
        self.discriminator.eval()
        self.generator.eval()
        if self.cycle_gan:
            self.c_discriminator.eval()
            self.c_generator.eval()
        labels, images = next(val_iter)
 
        #for pytorch input
        images = torch.tensor(images).to(self.device)
        images = images.permute(0,3,1,2)
        labels = torch.tensor(labels).to(self.device)

        with torch.no_grad():
            self.forward(images, labels, images, labels)
            self.d_backward(is_training=False)
            self.g_backward(is_training=False)

        print("Sample: d_loss: %.5f, g_loss: %.5f, l1_loss: %.5f" % (self.d_summary["d_loss"],
                                                                     self.g_summary["g_loss"],
                                                                     self.g_summary["l1_loss"]))

        fake_imgs = self.fake_B.permute(0,2,3,1).cpu().numpy()
        real_imgs = self.real_B.permute(0,2,3,1).cpu().numpy()

        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)

        model_sample_dir = self.get_save_dir(self.sample_dir)
        if not os.path.exists(model_sample_dir):
            os.makedirs(model_sample_dir)

        sample_img_path = os.path.join(model_sample_dir, "sample_%02d_%04d.png" % (epoch, step))
        misc.imsave(sample_img_path, merged_pair)

    def export_generator(self, save_dir, model_dir, model_name="gen_model.pth"):
        checkpoint_path = os.path.join(model_dir, 'model.pth')
        save_path = os.path.join(save_dir, model_name)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path)
            torch.save(ckpt['generator'], save_path)
            print("export generator model %s" % save_path)
        else:
            print("fail to export generator model %s" % save_path)

    def infer(self, source_obj, embedding_ids, model_dir, save_dir):
        # set test mode
        self.generator.eval()

        source_provider = InjectDataProvider(source_obj, rotate_range=self.rotate_range)

        if isinstance(embedding_ids, int) or len(embedding_ids) == 1:
            embedding_id = embedding_ids if isinstance(embedding_ids, int) else embedding_ids[0]
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, embedding_id)
        else:
            source_iter = source_provider.get_random_embedding_iter(self.batch_size, embedding_ids)

        self.restore_model(model_dir, is_training=False)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        for labels, source_imgs in source_iter:
            #for pytorch input
            source_imgs = torch.tensor(source_imgs).to(self.device)
            source_imgs = source_imgs.permute(0,3,1,2)
            labels = torch.tensor(labels).to(self.device)
            with torch.no_grad():
                self.forward(source_imgs, labels, source_imgs, labels)
            fake_imgs = self.fake_B.permute(0,2,3,1).cpu().numpy()
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def interpolate(self, source_obj, between, model_dir, save_dir, steps):
        # set test mode
        self.generator.eval()

        self.restore_model(model_dir, is_training=False)
        # new interpolated dimension
        new_x_dim = steps + 1
        alphas = np.linspace(0.0, 1.0, new_x_dim)

        def _interpolate_tensor(_tensor):
            """
            Compute the interpolated tensor here
            """

            x = _tensor[between[0]]
            y = _tensor[between[1]]

            interpolated = list()
            for alpha in alphas:
                interpolated.append((x * (1. - alpha) + alpha * y).view(1,-1))

            interpolated = torch.cat(interpolated, dim=0)
            return interpolated

        def filter_embedding_vars(var):
            var_name = var[0]
            if var_name.find("embedding") != -1:
                return True
            #For ConditionalInstanceNorm
            if self.inst_norm:
                if var_name.find("shift") != -1 or var_name.find("scale") != -1:
                    return True
            #For ConditionalBatchNorm
            if self.g_norm_type == "cbn":
                if var_name.find("weight_bar") != -1:
                    return True
            return False

        embedding_vars = filter(filter_embedding_vars, self.generator.named_parameters())
        # here comes the hack, we overwrite the original tensor
        # with interpolated ones. Note, the shape might differ

        # this is to restore the embedding at the end
        embedding_snapshot = list()
        for e_var in embedding_vars:
            _e_var = e_var[1].clone()
            embedding_snapshot.append((e_var[0], _e_var))
            if e_var[0].find("weight_bar") != -1:
                #For ConditionalBatchNorm
                input_tensor = e_var[1].data.permute(1,0).clone()
            else:
                input_tensor = e_var[1].clone()
            output_tensor = _interpolate_tensor(input_tensor)
            if e_var[0].find("weight_bar") != -1:
                #For ConditionalBatchNorm
                output_tensor = output_tensor.permute(1,0)
                e_var[1].data[:,:output_tensor.shape[1]] = output_tensor
            else:
                e_var[1].data = output_tensor
            print("overwrite %s tensor" % e_var[0], "old_shape ->", _e_var.shape, "new shape ->", e_var[1].data.shape)

        source_provider = InjectDataProvider(source_obj, rotate_range=self.rotate_range)

        for step_idx in range(len(alphas)):
            alpha = alphas[step_idx]
            print("interpolate %d -> %.4f + %d -> %.4f" % (between[0], 1. - alpha, between[1], alpha))
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, 0)
            batch_buffer = list()
            count = 0
            for _, source_imgs in source_iter:
                count += 1
                labels = [step_idx] * self.batch_size
                #for pytorch input
                source_imgs = torch.tensor(source_imgs).to(self.device)
                source_imgs = source_imgs.permute(0,3,1,2)
                labels = torch.tensor(labels).to(self.device)
                with torch.no_grad():
                    self.forward(source_imgs, labels, source_imgs, labels)
                generated = self.fake_B.permute(0,2,3,1).cpu().numpy()
                merged_fake_images = merge(scale_back(generated), [self.batch_size, 1])
                batch_buffer.append(merged_fake_images)
            if len(batch_buffer):
                save_concat_images(batch_buffer,
                                   os.path.join(save_dir, "frame_%02d_%02d_step_%02d.png" % (
                                       between[0], between[1], step_idx)))

        # restore the embedding variables
        print("restore embedding values")
        for e_var in embedding_snapshot:
            for name, param in self.generator.named_parameters():
                if name == e_var[0]:
                    param.data = e_var[1].data
            
    def train(self, lr=0.0002, epoch=100, schedule=10, resume=True, flip_labels=False,
              freeze_encoder=False, fine_tune=None, sample_steps=50, checkpoint_steps=500,
              ignore_label=None):
        # set train mode
        self.discriminator.train()
        self.generator.train()
        if self.cycle_gan:
            self.c_discriminator.train()
            self.c_generator.train()
        if freeze_encoder:
            self.freeze_encoder() 

        if self.cycle_gan:
            self.d_optimizer = torch.optim.Adam(itertools.chain(self.discriminator.parameters(), 
                                                                self.c_discriminator.parameters()),
                                                lr=lr, betas=(0.5, 0.999))
            self.g_optimizer = torch.optim.Adam(itertools.chain(self.generator.parameters(), 
                                                                self.c_generator.parameters()),
                                                lr=lr, betas=(0.5, 0.999))
        else:
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
            self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        # filter by one type of labels
        data_provider = TrainDataProvider(self.data_dir, filter_by=fine_tune, ignore_label=ignore_label, rotate_range=self.rotate_range)
        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val_iter(self.batch_size)

        train_log_dir = self.get_save_dir(self.log_dir)
        summary_writer = SummaryWriter(train_log_dir)

        if resume:
            model_dir = self.get_save_dir(self.checkpoint_dir)
            self.restore_model(model_dir)

        current_lr = lr
        counter = 0
        start_time = time.time()

        for ei in range(epoch):
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            if (ei + 1) % schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                counter += 1
                labels, batch_images = batch
                shuffled_ids = labels[:]

                if flip_labels:
                    np.random.shuffle(shuffled_ids)

                #for pytorch input
                batch_images = torch.tensor(batch_images).to(self.device)
                batch_images = batch_images.permute(0,3,1,2)
                labels = torch.tensor(labels).to(self.device)
                shuffled_ids = torch.tensor(shuffled_ids).to(self.device)

                self.forward(batch_images, labels, batch_images, shuffled_ids, no_target_source=flip_labels)
                # Optimize D
                self.d_optimizer.zero_grad()
                self.d_backward()
                self.d_optimizer.step()
                # Optimize G
                self.g_optimizer.zero_grad()
                self.g_backward(retain_graph=flip_labels)
                self.g_optimizer.step()

                # magic move to Optimize G again
                # according to https://github.com/carpedm20/DCGAN-tensorflow
                self.g_optimizer.zero_grad()
                self.forward(batch_images, labels, batch_images, shuffled_ids, no_target_source=flip_labels)
                self.g_backward()
                self.g_optimizer.step()

                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, " \
                             "category_loss: %.5f, cheat_loss: %.5f, const_loss: %.5f, l1_loss: %.5f" \
                              %(ei, bid, total_batches, passed, 
                                self.d_summary["d_loss"], self.g_summary["g_loss"],
                                self.d_summary["category_loss"], self.g_summary["cheat_loss"], 
                                self.g_summary["const_loss"], self.g_summary["l1_loss"])
                if "tv_loss" in self.g_summary:
                    log_format += ", tv_loss: %.5f" %(self.g_summary["tv_loss"])
                if "c_g_loss" in self.g_summary:
                    log_format += ", c_g_loss: %.5f" %(self.g_summary["c_g_loss"])
                if "c_d_loss" in self.d_summary:
                    log_format += ", c_d_loss: %.5f" %(self.d_summary["c_d_loss"])
                print(log_format)

                for key in self.d_summary:
                    summary_writer.add_scalar('Discriminator/%s' %(key), self.d_summary[key], counter)
                for key in self.g_summary:
                    summary_writer.add_scalar('Generator/%s' %(key), self.g_summary[key], counter)

                if counter % sample_steps == 0:
                    # sample the current model states with val data
                    self.validate_model(val_batch_iter, ei, counter)

                    # set train mode again
                    self.discriminator.train()
                    self.generator.train()
                    if self.cycle_gan:
                        self.c_discriminator.train()
                        self.c_generator.train()
                    if freeze_encoder:
                        self.freeze_encoder() 

                if counter % checkpoint_steps == 0:
                    print("Checkpoint: save checkpoint step %d" % counter)
                    self.checkpoint()
        # save the last checkpoint
        print("Checkpoint: last checkpoint step %d" % counter)
        self.checkpoint()