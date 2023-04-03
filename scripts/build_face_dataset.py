#!/usr/bin/env python3
import argparse
import ai_old.constants as c
import os
from PIL import Image
import shutil

# TODO: add support for mixing celeba, adience, etc. into ffhq train data
# TODO: filter images with sunglasses
# TODO: remove background

# 32 images we save at every eval
SAMPLES = [
    # young men
    'real-male/x/0898856126.png',
    'real-male/x/1066642604.png',
    'real-male/x/2427025053.png',
    'real-male/x/0590337936.png',
    'itw-256/x/00012.png',
    'itw-256/x/00083.png',
    'itw-256/x/00145.png',
    'itw-256/x/00198.png',

    # young women
    'itw-256/x/00001.png',
    'itw-256/x/00099.png',
    'celeba-512/imgs/123.jpg',
    'celeba-512/imgs/1230.jpg',
    'celeba-512/imgs/1231.jpg',
    'celeba-512/imgs/1238.jpg',
    'celeba-512/imgs/12301.jpg',
    'celeba-512/imgs/12310.jpg',

    # children
    'itw-256/x/00000.png',
    'itw-256/x/00003.png',
    'itw-256/x/00012.png',
    'itw-256/x/00086.png',
    'itw-256/x/00090.png',
    'itw-256/x/00225.png',
    'itw-256/x/00286.png',
    'itw-256/x/00301.png',

    # middle-aged
    'real-male/x/0272768511.png',
    'itw-256/x/00045.png',
    'itw-256/x/00081.png',
    'itw-256/x/00082.png',
    'celeba-512/imgs/12316.jpg',
    'celeba-512/imgs/12326.jpg',
    'itw-256/x/00206.png',
    'itw-256/x/00257.png',
]
N_VAL_FFHQ = 256
N_VAL_CELEBA = 256 - len(SAMPLES)
FFHQ_PATH = os.path.join(c.ASI_DATASETS_PATH, 'itw-256/x')
CELEBA_PATH = os.path.join(c.ASI_DATASETS_PATH, 'celeba-512/imgs')


def parse_args():
    parser = argparse.ArgumentParser(description='build face dataset')
    parser.add_argument('--size', type=int, required=True,
        help='64, 128, 256')
    return parser.parse_args()


def run(args):
    assert args.size in [64, 128, 256, 512]
    dir = 'face-{}'.format(args.size)
    base_path = os.path.join(c.ASI_DATASETS_PATH, dir)
    if os.path.exists(base_path):
        print('[WARNING] path exists: ' + base_path)
        _ = input('press enter to delete/recreate or ctrl-c to cancel... ')
        shutil.rmtree(base_path)

    print('loading ffhq paths')
    all_ffhq = []
    for filename in sorted(os.listdir(FFHQ_PATH)):
        all_ffhq.append(os.path.join(FFHQ_PATH, filename))

    print('loading celeba paths')
    all_celeba = []
    for filename in sorted(os.listdir(CELEBA_PATH)):
        all_celeba.append(os.path.join(CELEBA_PATH, filename))

    # val
    val_path = os.path.join(base_path, 'val')
    os.makedirs(val_path)
    val = [os.path.join(c.ASI_DATASETS_PATH, x) for x in SAMPLES]
    i = 0
    print('adding ffhq to val')
    while len(val) < N_VAL_FFHQ + len(SAMPLES):
        if all_ffhq[i] not in val:
            val.append(all_ffhq[i])
        i += 1
    i = 0
    print('adding celeba to val')
    while len(val) < N_VAL_CELEBA + N_VAL_FFHQ + len(SAMPLES):
        if all_celeba[i] not in val:
            val.append(all_celeba[i])
        i += 1
    print('writing val to disk')
    for i, path in enumerate(val):
        print('{}/{}'.format(i, len(val)))
        im = Image.open(path)
        w, h = im.size
        assert w == h
        if w != args.size:
            im = im.resize((args.size, args.size), Image.LANCZOS)
        im.save(os.path.join(val_path, f'{i:06}.png'), 'PNG')
    val = set(val)

    # train
    train_path = os.path.join(base_path, 'train')
    os.makedirs(train_path)
    i = 0
    print('writing train to disk')
    for path in all_ffhq:
        print('{}/{}'.format(i, len(all_ffhq)))
        if path not in val:
            im = Image.open(path)
            w, h = im.size
            assert w == h
            if w != args.size:
                im = im.resize((args.size, args.size), Image.LANCZOS)
            im.save(os.path.join(train_path, f'{i:06}.png'), 'PNG')
        i += 1


if __name__ == '__main__':
    args = parse_args()
    run(args)
