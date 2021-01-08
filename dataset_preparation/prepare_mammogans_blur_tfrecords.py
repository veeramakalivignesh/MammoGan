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

"""Create a tfrecords for celeba128x128 training. """

import zipfile
import tqdm
from defaults import get_cfg_defaults
import sys
import logging
from net import *
import numpy as np
import random
import argparse
import os
import tensorflow as tf
import imageio
from PIL import Image
from PIL import ImageFilter

def invert1(img):
    siz = int(img.shape[1]/2)
    if(img[:,:siz].mean()<img[:,siz:].mean()):
        return img[:,::-1]
    return img

def prepare_celeba(cfg, logger, train=True):
    if train:
        directory = os.path.dirname(cfg.DATASET.PATH)
    else:
        directory = os.path.dirname(cfg.DATASET.PATH_TEST)

    # with open("/data/datasets/CelebA/Eval/list_eval_partition.txt") as f:
    #     lineList = f.readlines()
    # lineList = [x[:-1].split(' ') for x in lineList]

    # split_map = {}
    # for x in lineList:
    #     split_map[int(x[0][:-4])] = int(x[1])

    os.makedirs(directory, exist_ok=True)

    # corrupted = [
    #     '195995.jpg',
    #     '131065.jpg',
    #     '118355.jpg',
    #     '080480.jpg',
    #     '039459.jpg',
    #     '153323.jpg',
    #     '011793.jpg',
    #     '156817.jpg',
    #     '121050.jpg',
    #     '198603.jpg',
    #     '041897.jpg',
    #     '131899.jpg',
    #     '048286.jpg',
    #     '179577.jpg',
    #     '024184.jpg',
    #     '016530.jpg',
    # ]

    def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
        # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
        if crop_w is None:
            crop_w = crop_h # the width and height after cropped
        h, w = x.shape[:2]
        crop_h,crop_w = x.shape[:2]
        return np.array(Image.fromarray(x).resize([resize_w, resize_w]))

    # archive = zipfile.ZipFile(os.path.join(directory, '//home/cse/btech/cs1170335/Code/Data/aiims_cc_512.tar.gz'), 'r')
    # names = archive.namelist()
    # names = [x for x in names if x[-4:] == '.jpg']
    names = []

    source_path = '/home/vikki/Desktop/BTP/ALAE/dataset_samples/mammogans_blur_samples'
    # source_path = '/home/cse/btech/cs1170335/Code/Data/aiims_cc_test_hd'
    for filename in tqdm.tqdm(os.listdir(source_path)):
        names.append((filename[:-4], filename))

    # if train:
    #     names = [x for x in names if split_map[int(x[:-4][-6:])] != 2]
    # else:
    #     names = [x for x in names if split_map[int(x[:-4][-6:])] == 2]

    count = len(names)
    print("Count: %d" % count)

    # names = [x for x in names if x[-10:] not in corrupted]

    random.seed(0)
    random.shuffle(names)

    folds = cfg.DATASET.PART_COUNT
    celeba_folds = [[] for _ in range(folds)]

    spread_identiteis_across_folds = False

    count_per_fold = count // folds
    for i in range(folds):
        celeba_folds[i] += names[i * count_per_fold: (i + 1) * count_per_fold]
    c = 0
    for i in range(folds):
        images = []
        for x in tqdm.tqdm(celeba_folds[i]):
            img_dir = os.path.join(source_path,x[1])
            imgfile = Image.open(img_dir)
            image = center_crop(imageio.imread(img_dir))
            image = np.array([image,image,image]).transpose((1,2,0))
            images.append((0,image.transpose((2, 0, 1))))
        
        print('Size of Images')
        print(len(images))

        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

        if train:
            part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)
        else:
            part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL, i)

        tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)

        random.shuffle(images)

        for label, image in images:
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        tfr_writer.close()

        for j in range(5):
            images_down = []
            
            # for label, image in tqdm.tqdm(images):
                # h = image.shape[1]
                # w = image.shape[2]
                # image = torch.tensor(np.asarray(image, dtype=np.float32)).view(1, 3, h, w)

                # image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8)

                # image_down = image_down.view(3, h // 2, w // 2).numpy()
                # images_down.append((label, image_down))

            for label, image in tqdm.tqdm(images):
                h = image.shape[1]
                w = image.shape[2]
                image = torch.tensor(np.asarray(image, dtype=np.float32)).view(1, 3, h, w)

                image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).to('cpu', torch.uint8)
                image_down = image_down.view(3, h // 2, w // 2).numpy()
                blurred_image = Image.fromarray(image_down.transpose(1,2,0).astype('uint8'), 'RGB').filter(ImageFilter.GaussianBlur(2))
                blurred_image = np.array(blurred_image).transpose(2,0,1)
                # Image.fromarray(image_down.transpose(1,2,0).astype('uint8'), 'RGB').save(str(c) + '.png')
                # c += 1
                # Image.fromarray(blurred_image.transpose(1,2,0).astype('uint8'), 'RGB').save(str(c) + '.png')
                # c += 1
                images_down.append((label, blurred_image))

            if train:
                part_path = cfg.DATASET.PATH % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)
            else:
                part_path = cfg.DATASET.PATH_TEST % (cfg.DATASET.MAX_RESOLUTION_LEVEL - j - 1, i)

            tfr_writer = tf.python_io.TFRecordWriter(part_path, tfr_opt)
            for label, image in images_down:
                # print(image.shape)
                # Image.fromarray(image.transpose((1,2,0)).astype('uint8'), 'RGB').save("/home/vikki/Desktop/BTP/ALAE/dataset_samples/blur_test"+str(j)+"/"+str(i)+".png")
                ex = tf.train.Example(features=tf.train.Features(feature={
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()]))}))
                tfr_writer.write(ex.SerializeToString())
            tfr_writer.close()

            images = images_down


def run():
    parser = argparse.ArgumentParser(description="ALAE. Prepare tfrecords for celeba128x128")
    parser.add_argument(
        "--config-file",
        default="configs/mammogans.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    prepare_celeba(cfg, logger, True)
    prepare_celeba(cfg, logger, False)


if __name__ == '__main__':
    run()