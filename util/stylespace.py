#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from external.e4e.models.stylegan2.model import Generator
from PIL import Image
import numpy as np


INDEX = [0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,10,11,12,12,13,14,14,15,16,16]


def load_model():
    path = '/home/asiu/data/models/stylegan/stylegan2-ffhq-config-f.pt'
    ckpt = torch.load(path)
    model = Generator(1024, 512, 8).to('cuda')
    model.load_state_dict(ckpt['g_ema'], strict=False)
    model = model.eval()
    # latent_avg = ckpt['latent_avg']
    return model


def visual(output, path):
    # output = (output + 1)/2
    # output = torch.clamp(output, 0, 1)
    # if output.shape[1] == 1:
    #     output = torch.cat([output, output, output], 1)
    # output = output[0].detach().cpu().permute(1,2,0).numpy()
    # output = (output*255).astype(np.uint8)
    # plt.imshow(output)
    # plt.show()
    output = (output * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(
        np.transpose(output, (1, 2, 0)),
        'RGB',
    ).save(path)


def conv_warper(layer, input, style, noise):
    # the conv should change
    conv = layer.conv
    batch, in_channel, height, width = input.shape

    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel,
        in_channel,
        conv.kernel_size,
        conv.kernel_size,
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch,
            conv.out_channel,
            in_channel,
            conv.kernel_size,
            conv.kernel_size,
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel,
            conv.out_channel,
            conv.kernel_size,
            conv.kernel_size,
        )
        out = F.conv_transpose2d(
            input,
            weight,
            padding=0,
            stride=2,
            groups=batch,
        )
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    out = layer.noise(out, noise=noise)
    out = layer.activate(out)

    return out

def decoder(G, stylespace, w, noise):
    # an decoder warper for G
    out = G.input(w)
    out = conv_warper(G.conv1, out, stylespace[0], noise[0])
    skip = G.to_rgb1(out, w[:, 1])

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, stylespace[i], noise=noise1)
        out = conv_warper(conv2, out, stylespace[i+1], noise=noise2)
        skip = to_rgb(out, w[:, i + 2], skip)

        i += 2

    image = skip

    return image

def encoder(G, seed):
    # an encoder warper for G
    stylespace = []
    w = G.style(seed)
    noise = [
        getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)
    ]
    inject_index = G.n_latent
    w = w.unsqueeze(1).repeat(1, inject_index, 1)
    stylespace.append(G.conv1.conv.modulation(w[:, 0]))

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        stylespace.append(conv1.conv.modulation(w[:, i]))
        stylespace.append(conv2.conv.modulation(w[:, i+1]))
        i += 2

    print(len(stylespace))
    for x in stylespace:
        print(x.shape)
    return stylespace, w, noise

if __name__ == '__main__':
    generator = load_model()
    seed = torch.randn(1,512).cuda()

    output, _ = generator([seed], False)
    visual(output[0], '/home/asiu/data/styleclip/tmp/0.png')

    # for 9_409 in the paper, you should have "stylespace[INDEX[9]][:, 409]"
    # the value to shift is hand-craft

    # eye
    stylespace, w, noise = encoder(generator, seed)
    stylespace[INDEX[9]][:, 409] += 10
    image = decoder(generator, stylespace, w, noise)
    visual(image[0], '/home/asiu/data/styleclip/tmp/1.png')

    # hair
    stylespace, w, noise = encoder(generator, seed)
    stylespace[INDEX[12]][:, 330] -= 50
    image = decoder(generator, stylespace, w, noise)
    visual(image[0], '/home/asiu/data/styleclip/tmp/2.png')

    # mouth
    stylespace, w, noise = encoder(generator, seed)
    stylespace[INDEX[6]][:, 259] -= 20
    image = decoder(generator, stylespace, w, noise)
    visual(image[0], '/home/asiu/data/styleclip/tmp/3.png')

    # lip
    stylespace, w, noise = encoder(generator, seed)
    stylespace[INDEX[15]][:, 45] -= 3
    image = decoder(generator, stylespace, w, noise)
    visual(image[0], '/home/asiu/data/styleclip/tmp/4.png')
