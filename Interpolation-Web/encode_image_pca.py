import torch.utils.data
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
from PIL import Image, ImageOps
import bimpy
import logging
import pickle as pkl
import matplotlib.pyplot as plt
import cv2

def make_square(img):
        ver = img.shape[0]
        hor = img.shape[1]
        ml = img[:,:hor//2].mean()
        mr = img[:,hor//2:].mean()
        if ml>mr:
            if hor<ver:
                arr = np.array(ver*[(ver-hor)*[0]])
                return np.concatenate((img,arr),1)
            else:
                return img[:,:ver]
        else:
            if hor<ver:
                arr = np.array(ver*[(ver-hor)*[0]])
                return np.concatenate((arr,img),1)
            else:
                return img[:,hor-ver:]

class Model_Web():
    def __init__(self, socketio, cfg):
        self.img_size = 256
        # self.img_size = 512
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

        with open("pca_result.pkl", "rb") as file:
            (self.comp,self.mean,self.std,_) = pkl.load(file)

        print ("Model Build Complete")

    def encode(self,x):
        Z, _ = self.model.encode(x, self.layer_count - 1, 1)
        Z = Z.repeat(1, self.model.mapping_fl.num_layers, 1)
        return Z

    def decode(self,x):
        layer_idx = torch.arange(2 * self.layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < self.model.truncation_cutoff, ones, ones)
        return self.model.decoder(x, self.layer_count - 1, 1, noise=True)

    def update_image(self,latents):
        with torch.no_grad():
            x_rec = self.decode(latents)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)
    
    def encode_image(self, file_path):
        img = Image.open(file_path)
        img = ImageOps.grayscale(img)
        img = np.asarray(img)
        if img.shape[0]!=img.shape[1]:
            img = make_square(img)
        img = img.astype('uint8')
        img = Image.fromarray(img)
        img.resize((self.img_size,self.img_size))
        img = np.asarray(img)
        img = np.array([img,img,img]).transpose((1,2,0))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]

        needed_resolution = self.model.decoder.layer_to_resolution[-1]
        while x.shape[2] > needed_resolution:
            x = F.avg_pool2d(x, 2, 2)
        if x.shape[2] != needed_resolution:
            x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

        latents_original = self.encode(x[None, ...].cuda())

        return latents_original 
    
    def get_gen_img(self,z):
        out = self.update_image(z)
        img_src = out.type(torch.long).clamp(0, 255).cpu().type(torch.uint8).numpy()
        im = Image.fromarray(img_src)
        im = im.resize((512,512),Image.BICUBIC)
        im.save('gen.png')

    def get_components(self,z):
        values = []
        i=0
        for w in self.comp:
            values.append((((z-self.mean)/self.std) * w).sum().item())
            i += 1
        return values
    
    def get_new_latents(self,z,ind,val):
        new_latents = (((z-self.mean)/self.std + val*self.comp[ind])*self.std + self.mean).float()
        return new_latents
    
    def resize(self, img, desired_size):
        im = img
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im.thumbnail(new_size, Image.ANTIALIAS)
    
        new_im = Image.new("L", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))

        return new_im