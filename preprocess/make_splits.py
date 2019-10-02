import os
import glob


ANNO_ROOT = '/home/weilians/data/floorplan_annotation/'
GA_ROOT = '/local-scratch/weilians/data/GA-Floorplan/'

OUTPUT_ROOT = GA_ROOT + 'preprocess/hrnet-data/'


def main():
  img_paths = sorted(glob.glob(OUTPUT_ROOT + 'images/*.png'))
  mask_paths = sorted(glob.glob(OUTPUT_ROOT + 'masks/*.png'))

  img_fnames = [os.path.basename(x) for x in img_paths]
  masks_fnames = [os.path.basename(x) for x in mask_paths]

  assert img_fnames == masks_fnames

  train_img_paths = img_paths[:20]
  train_mask_paths = mask_paths[:20]

  test_img_paths = img_paths[20:]
  test_mask_paths = mask_paths[20:]

  with open('train.lst', 'w') as f:
    for img_path, mask_path in zip(train_img_paths, train_mask_paths):
      f.write('%s %s\n' % (img_path, mask_path))

  with open('test.lst', 'w') as f:
    for img_path, mask_path in zip(test_img_paths, test_mask_paths):
      f.write('%s %s\n' % (img_path, mask_path))


if __name__ == '__main__':
  main()
