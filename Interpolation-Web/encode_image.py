import torch.utils.data
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from PIL import Image
import bimpy
import logging

# import tensorflow as tf
# import os
# from tensorflow.python.framework import meta_graph
import matplotlib.pyplot as plt
import cv2
# import random
# import pickle
# import sys
# import numpy as np

class Model_Web():
    def __init__(self, socketio, cfg):
        self.img_size = 256
        self.iterations = 0
        self.socketio = socketio
        self.update_progess = 10
        self.cfg = cfg
        
    def build_model(self):
        print ("Building Model .... ")
        
        torch.cuda.set_device(0)
        self.model = Model(
            startf=self.cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=self.cfg.MODEL.LAYER_COUNT,
            maxf=self.cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=self.cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=self.cfg.MODEL.TRUNCATIOM_PSI,
            truncation_cutoff=self.cfg.MODEL.TRUNCATIOM_CUTOFF,
            mapping_layers=self.cfg.MODEL.MAPPING_LAYERS,
            channels=self.cfg.MODEL.CHANNELS,
            generator=self.cfg.MODEL.GENERATOR,
            encoder=self.cfg.MODEL.ENCODER)
        self.model.cuda(0)
        self.model.eval()
        self.model.requires_grad_(False)

        self.decoder = self.model.decoder
        self.encoder = self.model.encoder
        self.mapping_tl = self.model.mapping_tl
        self.mapping_fl = self.model.mapping_fl
        self.dlatent_avg = self.model.dlatent_avg

        logger = logging.getLogger("logger")
        logger.setLevel(logging.DEBUG)

        logger.info("Trainable parameters generator:")
        count_parameters(self.decoder)

        logger.info("Trainable parameters discriminator:")
        count_parameters(self.encoder)

        model_dict = {
            'discriminator_s': self.encoder,
            'generator_s': self.decoder,
            'mapping_tl_s': self.mapping_tl,
            'mapping_fl_s': self.mapping_fl,
            'dlatent_avg': self.dlatent_avg
        }

        checkpointer = Checkpointer(self.cfg,
                                    model_dict,
                                    {},
                                    logger=logger,
                                    save=False)

        self.extra_checkpoint_data = checkpointer.load()

        self.model.eval()

        self.layer_count = self.cfg.MODEL.LAYER_COUNT
        print ("Model Build Complete")

    def encode(self,x):
        Z, _ = self.model.encode(x, self.layer_count - 1, 1)
        Z = Z.repeat(1, self.model.mapping_fl.num_layers, 1)
        return Z

    def decode(self,x):
        layer_idx = torch.arange(2 * self.layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < self.model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return self.model.decoder(x, self.layer_count - 1, 1, noise=True)

    def update_image(self,latents):
        with torch.no_grad():
            x_rec = self.decode(latents)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)
    
    def encode_image(self, left_file_path, right_file_path):
        img = np.asarray(Image.open(left_file_path))
        imgl = np.array([img,img,img]).transpose((1,2,0))
        img = np.asarray(Image.open(right_file_path))
        imgr = np.array([img,img,img]).transpose((1,2,0))
        if imgl.shape[2] == 4:
            imgl = imgl[:, :, :3]
        iml = imgl.transpose((2, 0, 1))
        xl = torch.tensor(np.asarray(iml, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if xl.shape[0] == 4:
            xl = xl[:3]

        needed_resolution = self.model.decoder.layer_to_resolution[-1]
        print(needed_resolution)
        while xl.shape[2] > needed_resolution:
            xl = F.avg_pool2d(xl, 2, 2)
        if xl.shape[2] != needed_resolution:
            xl = F.adaptive_avg_pool2d(xl, (needed_resolution, needed_resolution))

        latents_original_left = self.encode(xl[None, ...].cuda())

        if imgr.shape[2] == 4:
            imgr = imgr[:, :, :3]
        imr = imgr.transpose((2, 0, 1))
        xr = torch.tensor(np.asarray(imr, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if xr.shape[0] == 4:
            xr = xr[:3]

        needed_resolution = self.model.decoder.layer_to_resolution[-1]
        print(needed_resolution)
        while xr.shape[2] > needed_resolution:
            xr = F.avg_pool2d(xr, 2, 2)
        if xr.shape[2] != needed_resolution:
            xr = F.adaptive_avg_pool2d(xr, (needed_resolution, needed_resolution))

        latents_original_right = self.encode(xr[None, ...].cuda())

        return latents_original_left, latents_original_right 
    
    def get_gen_img(self, left_z, right_z, alpha):
        alpha = alpha/100.0
        new_latents = alpha*right_z + (1-alpha)*left_z
        out = self.update_image(new_latents)
        print(alpha)
        img_src = out.type(torch.long).clamp(0, 255).cpu().type(torch.uint8).numpy()
        im = Image.fromarray(img_src)
        im.save('gen.png')