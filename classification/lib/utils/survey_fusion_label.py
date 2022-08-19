from json.tool import main
import cv2
import numpy
import os


root = '/data/huangyx/data/HSICityV2/'
def load(name):
    coarse_label_path = os.path.join(root, 'val', name + '_gray.png')
    fine_label_path = os.path.join(root, 'test', name + '_gray.png')
    cls_res_path = os.path.join(root, 'val', name + '_gray.png')
    

def main():
    lst_path = '~/code/HSI2seg/data/list/hsicity2/val.lst'
    lst = [l.strip() for l in open(lst_path)]
    for i in lst:


if __name__=='main':
    main()