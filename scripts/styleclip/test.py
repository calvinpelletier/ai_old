#!/usr/bin/env python3
from ai_old.util.styleclip.inference import StyleClip
from PIL import Image

sc = StyleClip()
sc.neutral = 'hair'
sc.target = 'hi-top fade hair'
sc.get_dt2()
print(sc.beta)
alphas = [0, 2, 4, 8, 16]
betas = [0.27, 0.23, 0.19, 0.15, 0.11]
imsize = 1024
canvas = Image.new(
    'RGB',
    (imsize * len(alphas), imsize * len(betas)),
    'black',
)
for x in range(len(alphas)):
    for y in range(len(betas)):
        sc.M.alpha = [alphas[x]]
        sc.beta = betas[y]
        canvas.paste(
            Image.fromarray(sc.get_img()),
            (x * imsize, y * imsize),
        )
canvas.save('/home/asiu/data/styleclip/tmp/0.png')
