import argparse
import os
from dataset import HSICity2
from PIL import Image
import cv2

labels = HSICity2.hsicity_label

def get_palette_hsicity(n):
  palette = [0] * (n * 3)
  for j in range(0, len(labels)):
    palette[j * 3] = labels[j][1][0]
    palette[j * 3 + 1] = labels[j][1][1]
    palette[j * 3 + 2] = labels[j][1][2]
  return palette

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str)
parser.add_argument('--outdir', type=str)

config = parser.parse_args()

palette = get_palette_hsicity(256)

for ret in os.listdir(config.indir):
  img = os.path.join(config.indir, ret)
  pred = Image.open(img).convert('P')
  pred.putpalette(palette)
  pred = pred.resize((1889, 1422))
  pred.save(f'{config.outdir}/{ret[:-4]}_vis.png')