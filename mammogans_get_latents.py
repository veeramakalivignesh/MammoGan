# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.utils.data
from torchvision.utils import save_image
import random
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
import tqdm
from PIL import Image
import pickle as pkl


lreq.use_implicit_lreq.set(True)

def place(canvas, image, x, y):
    im_size = image.shape[2]
    if len(image.shape) == 4:
        image = image[0]
    canvas[:, y: y + im_size, x: x + im_size] = image * 0.5 + 0.5


def save_sample(model, sample, i):
    os.makedirs('results', exist_ok=True)

    with torch.no_grad():
        model.eval()
        x_rec = model.generate(model.generator.layer_count - 1, 1, z=sample)

        def save_pic(x_rec):
            resultsample = x_rec * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample,
                       'sample_%i_lr.png' % i, nrow=16)

        save_pic(x_rec)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    # print("hi")
    # cfg.OUTPUT_DIR = "mammogans_blur_result"
    # print(cfg.OUTPUT_DIR)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model.cuda(0)
    model.eval()
    model.requires_grad_(False)
    # cfg.OUTPUT_DIR = "mammogans_blur_result"
    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    path = cfg.DATASET.SAMPLES_PATH
    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)

    paths = list(os.listdir(path))

    paths = sorted(paths)
    random.seed(1)
    random.shuffle(paths)

    def make(paths):
        latent_set = []
        with torch.no_grad():
            loss = 0
            for filename in paths:
                img = np.asarray(Image.open(path + '/' + filename))
                if(img.shape==(28,28)):
                    img = np.pad(img,(2,2))
                    img = np.array([img,img,img]).transpose((1,2,0))
                flag = False
                if(img.shape==(512,512)):
                    # if(img[:,:256].mean()<img[:,256:].mean()):
                    #     flag = True
                    #     img = img[:,::-1]
                    img = np.array([img,img,img]).transpose((1,2,0))
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
                if x.shape[0] == 4:
                    x = x[:3]
                factor = x.shape[2] // im_size
                if factor != 1:
                    x = torch.nn.functional.avg_pool2d(x[None, ...], factor, factor)[0]
                assert x.shape[2] == im_size
                latents = encode(x[None, ...].cuda())
                latent_set.append(latents)
        return latent_set

    def chunker_list(seq, n):
        return [seq[i * n:(i + 1) * n] for i in range((len(seq) + n - 1) // n)]

    paths = chunker_list(paths, 8 * 3)

    for i, chunk in enumerate(paths):
        latent_list = make(chunk)
        if i==0:
            latent_set = torch.cat(latent_list, dim=0)
        else:
            temp = torch.cat(latent_list, dim=0)
            latent_set = torch.cat([temp,latent_set], dim=0)
        print(latent_set.shape)
    with open("latents.pkl", "wb") as fout:
        pkl.dump(latent_set, fout, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-figure-reconstructions-paged', default_config='configs/mammogans_hd.yaml',
        world_size=gpu_count, write_log=False)
