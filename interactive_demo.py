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
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq

from PIL import Image
import bimpy


lreq.use_implicit_lreq.set(True)


indices = [0]

labels = ["interpolation constant"]


def sample(cfg, logger):
    torch.cuda.set_device(0)
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
        layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        # x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    path = 'dataset_samples/mammogans_blur_samples/'

    paths = list(os.listdir(path))
    paths.sort()
    paths_backup = paths[:]
    randomize = bimpy.Bool(False)
    current_file = bimpy.String("")
    current_file1 = bimpy.String("")

    ctx = bimpy.Context()

    attribute_values = [bimpy.Float(0) for i in indices]

    W = [torch.tensor(np.load("principal_directions/direction_%d.npy" % i), dtype=torch.float32) for i in indices]

    rnd = np.random.RandomState(5)

    def loadNext(flag):
        img = np.asarray(Image.open(path + '/' + paths[0]))
        if flag:
            current_file.value = paths[0]
        else:
            current_file1.value = paths[0]
        paths.pop(0)
        if len(paths) == 0:
            paths.extend(paths_backup)

        img = np.array([img,img,img]).transpose((1,2,0))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]

        needed_resolution = model.decoder.layer_to_resolution[-1]
        # needed_resolution = 512
        print(needed_resolution)
        while x.shape[2] > needed_resolution:
            x = F.avg_pool2d(x, 2, 2)
        if x.shape[2] != needed_resolution:
            x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

        img_src = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()

        latents_original = encode(x[None, ...].cuda())
        latents = latents_original[0, 0].clone()
        latents -= model.dlatent_avg.buff.data[0]

        for v, w in zip(attribute_values, W):
            v.value = (latents * w).sum()

        for v, w in zip(attribute_values, W):
            latents = latents - v.value * w

        return latents, latents_original, img_src

    def loadRandom():
        latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
        lat = torch.tensor(latents).float().cuda()
        dlat = mapping_fl(lat)
        layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        dlat = torch.lerp(model.dlatent_avg.buff.data, dlat, coefs)
        x = decode(dlat)[0]
        img_src = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()
        latents_original = dlat
        latents = latents_original[0, 0].clone()
        latents -= model.dlatent_avg.buff.data[0]

        for v, w in zip(attribute_values, W):
            v.value = (latents * w).sum()

        for v, w in zip(attribute_values, W):
            latents = latents - v.value * w

        return latents, latents_original, img_src

    latents, latents_original, img_src = loadNext(True)
    latents1, latents_original1, img_src1 = loadNext(False)

    ctx.init(1800, 1600, "Styles")

    def update_image(latents):
        with torch.no_grad():
            # w = w + model.dlatent_avg.buff.data[0]
            # w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1)

            # layer_idx = torch.arange(model.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
            # cur_layers = (7 + 1) * 2
            # mixing_cutoff = cur_layers
            # styles = torch.where(layer_idx < mixing_cutoff, w, latents_original)

            x_rec = decode(latents)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
    im = update_image(latents_original)
    im1 = update_image(latents_original1)
    im2 = update_image(latents_original1)
    print(im.shape)
    im = bimpy.Image(im)
    im1 = bimpy.Image(im1)
    im2 = bimpy.Image(im2)

    display_original = True

    seed = 0

    while not ctx.should_close():
        with ctx:
            alpha = attribute_values[0].value
            new_latents = alpha*latents_original + (1-alpha)*latents_original1
            # print(attribute_values[0].value)

            if display_original:
                im = bimpy.Image(img_src)
                im1 = bimpy.Image(img_src1)
                im2 = bimpy.Image(img_src1)
            else:
                im2 = bimpy.Image(update_image(new_latents))

            bimpy.begin("Principal directions")
            bimpy.columns(3)
            bimpy.set_column_width(0, im_size + 20)
            bimpy.image(im)
            bimpy.image(im1)
            bimpy.next_column()
            bimpy.image(im2)
            bimpy.next_column()

            for v, label in zip(attribute_values, labels):
                bimpy.slider_float(label, v, 0.0, 1.0)

            bimpy.checkbox("Randomize noise", randomize)

            if randomize.value:
                seed += 1

            torch.manual_seed(seed)

            if bimpy.button('Display Reconstruction'):
                display_original = False

            if bimpy.input_text("Mammogram 1", current_file, 64) and os.path.exists(path + '/' + current_file.value):
                paths.insert(0, current_file.value)
                latents, latents_original, img_src = loadNext(True)
            
            if bimpy.input_text("Mammogram 2", current_file1, 64) and os.path.exists(path + '/' + current_file1.value):
                paths.insert(0, current_file1.value)
                latents1, latents_original1, img_src1 = loadNext(False)

            bimpy.end()


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-interactive', default_config='configs/mammogans_hd.yaml',
        world_size=gpu_count, write_log=False)
