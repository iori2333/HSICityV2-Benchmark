# -*- coding:utf-8 -*-
from utils.utils import confidence_label_sort, confidence_label_softmax
from .base_dataset import BaseDataset
import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

import sys
print(sys.path)


class hsicity(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=10,
                 multi_scale=True,
                 flip=True,
                 ignore_label=0,
                 base_size=1379,
                 crop_size=(1773, 1379),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 use_rgb=False):
        super(hsicity, self).__init__(ignore_label, base_size,
                                      crop_size, downsample_rate, scale_factor, )
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = torch.FloatTensor([
        0.6666, 0.9572, 0.7315, 0.8643, 0.8280, 1.9843, 0.9995, 0.8465, 0.6949,
        1.0520, 0.7580, 0.9479, 0.9912, 0.6867, 0.8538, 0.7671, 1.2733, 1.9843,
        1.1129]).cuda()
        self.label_mapping = {i: i for i in range(self.num_classes)}

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        self.use_rgb = use_rgb
        self.img_list = [line.strip().split() for line in open(list_path)]
        self.files = self.read_files()

        self.hsicity_label = [
            (0, (128, 64, 128)),  # road
            (1, (244, 35, 232)),  # sidewalk
            (2, (70, 70, 70)),  # building
            (3, (102, 102, 156)),  # wall
            (4, (190, 153, 153)),  # fence
            (5, (153, 153, 153)),  # pole
            (6, (250, 170, 30)),  # traffic light
            (7, (220, 220, 0)),  # traffic sign
            (8, (107, 142, 35)),  # vegetation
            (9, (152, 251, 152)),  # terrain
            (10, (70, 130, 180)),  # sky
            (11, (220, 20, 60)),  # person
            (12, (255, 0, 0)),  # rider
            (13, (0, 0, 142)),  # car
            (14, (0, 0, 70)),  # truck
            (15, (0, 60, 100)),  # bus
            (16, (0, 80, 100)),  # train
            (17, (0, 0, 230)),  # motorcycle
            (18, (119, 11, 32)),  # bicycle
        ]

    def read_files(self):
        files = []
        if 'test_' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                names = item[0].split('/')
                pre = names[0]
                name = names[1]
                image_path = os.path.join(pre, name + '.hsd')
                rgb_image_path = os.path.join(pre, 'rgb' + name + '.jpg')
                label_path = os.path.join(pre, 'rgb' + name + '_gray.png')
                files.append({
                    "img": image_path,
                    "rgbimg": rgb_image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        if self.use_rgb:
            image = cv2.imread(os.path.join(self.root, item["rgbimg"]),
                        cv2.IMREAD_COLOR)
            image = image[:, :, ::-1]  # bgr->rgb
        else:
            image = self.read_HSD(os.path.join(self.root, item["img"]))
        #image = image.transpose(1, 0, 2)[:, ::-1, :]

        size = image.shape

        label = cv2.imread(os.path.join(self.root, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        #label = label.transpose(1, 0)[:, ::-1]  # covert�?放label
        # label[label == 255] = -1
        
        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip,
                                       self.center_crop_test)
        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:,
                                                          :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = F.upsample(preds, (ori_height, ori_width),
                               mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette_hsicity(self, n):
        palette = [0] * (n * 3)
        for j in range(0, len(self.hsicity_label)):
            palette[j * 3] = self.hsicity_label[j][1][0]
            palette[j * 3 + 1] = self.hsicity_label[j][1][1]
            palette[j * 3 + 2] = self.hsicity_label[j][1][2]
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette_hsicity(256)
        preds = preds.cpu().numpy().copy()
        # np.save(os.path.join(sv_path, name[0] + '.npy'), preds)
        preds = np.asarray(np.argmax(preds, axis=2), dtype=np.uint8) 
        # preds = confidence_label_softmax(preds, threshold=0.7)

        pred = self.convert_label(preds, inverse=False)
        save_img = Image.fromarray(pred)
        save_img.putpalette(palette)
        save_img.save(os.path.join(sv_path, name[0] + 'vis.png'))

        preds = preds.squeeze()
        save_img = Image.fromarray(preds)
        save_img.save(os.path.join(sv_path, name[0] + '.png'))

    def save_pred_gray(self, preds, sv_path, name):
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=2), dtype=np.uint8)

        preds = preds.squeeze()
        save_img = Image.fromarray(preds)
        save_img.save(os.path.join(sv_path, name + '.png'))

    def read_HSD(self, filename):
        data = np.fromfile('%s' % filename, dtype=np.int32)
        height = data[0]
        width = data[1]
        SR = data[2]
        D = data[3]

        data = np.fromfile('%s' % filename, dtype=np.float32)
        a = 7
        average = data[a:a + SR]
        a = a + SR
        coeff = data[a:a + D * SR].reshape((D, SR))
        a = a + D * SR
        scoredata = data[a:a + height * width * D].reshape((height * width, D))

        temp = np.dot(scoredata, coeff)
        data = (temp + average).reshape((height, width, SR))

        return data
