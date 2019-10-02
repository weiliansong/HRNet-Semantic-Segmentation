import os
import glob
import json

import cv2
import numpy as np

from PIL import Image, ImageOps


ANNO_ROOT = '/home/weilians/data/floorplan_annotation/'
GA_ROOT = '/local-scratch/weilians/data/GA-Floorplan/'

OUTPUT_ROOT = GA_ROOT + 'preprocess/hrnet-data/'


def process_annotation(anno):
  for floorplan_key in anno.keys():
    # skip this floorplan if it is not annotated
    if not len(anno[floorplan_key]['regions']):
      continue

    floorplan_id = floorplan_key.strip().split('.')[0]

    # load up floorplan image
    img_path = ANNO_ROOT + 'floorplan/%s.jpg' % floorplan_id
    img = Image.open(img_path)
    img = np.array(img, dtype=np.uint8)
    img_height, img_width, _ = img.shape

    # this will be our label mask
    mask = np.zeros_like(img)

    # parse the annotated shapes and render polygons
    for region in anno[floorplan_key]['regions']:
      polygon_x = region['shape_attributes']['all_points_x']
      polygon_y = region['shape_attributes']['all_points_y']
      polygon_pts = np.array([list(zip(polygon_x, polygon_y)),])

      cv2.fillPoly(mask, polygon_pts, color=[255, 255, 255])

    # rotate the image and mask so it's up-right
    if img_width > img_height:
      img = np.rot90(img)
      mask = np.rot90(mask)

    # crop the image vertically if it is wider than 1024
    if img_width > 1024:
      start = (img_width - 1024) // 2
      end = img_width - 1024 - start

      img = img[:, start:end, :]
      mask = mask[:, start:end, :]

    # pad the image so it's 2048 tall
    _img = np.zeros([2048, 1024, 3], dtype=img.dtype)
    _mask = np.zeros([2048, 1024, 3], dtype=mask.dtype)

    _img[0:img.shape[0], 0:img.shape[1], :] = img
    _mask[0:mask.shape[0], 0:mask.shape[1], :] = mask

    img = _img.copy()
    mask = _mask.copy()

    # save out the image and mask pair
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)

    img.save(OUTPUT_ROOT + 'images/%s.png' % floorplan_id)
    mask.save(OUTPUT_ROOT + 'masks/%s.png' % floorplan_id)


def main():
  # make necessary directories
  if not os.path.exists(OUTPUT_ROOT + 'images/'):
    os.makedirs(OUTPUT_ROOT + 'images/')

  if not os.path.exists(OUTPUT_ROOT + 'masks/'):
    os.makedirs(OUTPUT_ROOT + 'masks/')

  # grab all the JSON annotations
  with open(GA_ROOT + 'room_annotations/wall_anno_amin.json', 'r') as f:
    amin_anno = json.load(f)

  with open(GA_ROOT + 'room_annotations/wall_anno_william.json', 'r') as f:
    william_anno = json.load(f)['_via_img_metadata']

  process_annotation(amin_anno)
  process_annotation(william_anno)


if __name__ == '__main__':
  main()
