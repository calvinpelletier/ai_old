#!/usr/bin/env python3
from PIL import Image
import os


def run(dir, n, out):
    count = 0
    for filename in os.listdir(dir):
        id, gender = filename.split('.')[0].split('_')
        if gender == 'm':
            print(count)
            canvas = Image.new('RGB', (256 * 2, 256), 'black')
            im1 = Image.open(os.path.join(dir, f'{id}_m.png'))
            im2 = Image.open(os.path.join(dir, f'{id}_f.png'))
            canvas.paste(im1, (0, 0))
            canvas.paste(im2, (256, 0))
            canvas.save(os.path.join(out, f'{id}.png'))
            count += 1
            if count >= n:
                break


if __name__ == '__main__':
    run(
        '/home/calvin/data/asi/z-exploration/y/0/ims',
        1000,
        '/home/calvin/projects/asi/devserver/static/img/labelee',
    )
    run(
        '/home/calvin/data/asi/z-exploration/y/1/ims',
        1000,
        '/home/calvin/projects/asi/devserver/static/img/labelee',
    )
