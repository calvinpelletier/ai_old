#!/usr/bin/env python3
import math
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torch
import numpy as np
import ai_old.constants as c

USE_PIXEL_CENTER = False
BOUNDARY_MASK_REL_SIZE = 0.05  # rel to solo width
BOUNDARY_MASK_INNER_REL_OVERLAP = 0.1  # rel to BOUNDARY_MASK_REL_SIZE


def solo_aligned_to_fam_final(
        solo_aligned,
        quad,
        fam_og,
        debug=False,
):
    w, h = fam_og.size
    print(quad)
    quad = [int(round(x)) for x in list(quad)]
    print(quad)

    # inverse bilinear transformation
    solo_unaligned, solo_unaligned_mask = _inverse_align(
        solo_aligned,
        quad,
        w, h,
        debug=debug,
    )

    # reinsert into the fam photo
    fam_final = _reinsert(fam_og, solo_unaligned, solo_unaligned_mask)

    if debug:
        return fam_final, solo_unaligned
    return fam_final


def solo_aligned_to_fam_final_and_inpaint(
        solo_aligned,
        quad,
        fam_og,
        inpainter,
        debug=False,
):
    w, h = fam_og.size
    quad = [int(round(x)) for x in list(quad)]

    if debug:
        _save(solo_aligned, 'solo_aligned')
        fam_og_quad = _draw_quad(fam_og, quad)
        _save(fam_og_quad, 'fam_og_quad')

    # inverse bilinear transformation
    solo_unaligned, solo_unaligned_mask, boundary_mask = _inverse_align(
        solo_aligned,
        quad,
        w, h,
        needs_boundary_mask=True,
    )

    # reinsert into the fam photo
    fam_reinserted = _reinsert(fam_og, solo_unaligned, solo_unaligned_mask)

    if debug:
        _save(fam_reinserted, 'fam_reinserted')

        bm_w, bm_h = boundary_mask.size
        black = Image.new(fam_reinserted.mode, (bm_w, bm_h), (0, 0, 0))
        debug_boundary_mask = _reinsert(fam_reinserted, black, boundary_mask)
        debug_boundary_mask = _draw_quad(debug_boundary_mask, quad)
        _save(debug_boundary_mask, 'debug_boundary_mask')

    # mask and inpaint the boundary between the altered and unaltered regions
    fam_final = _mask_and_inpaint(inpainter, fam_reinserted, boundary_mask)

    output_img = Image.fromarray(
        np.transpose(np.uint8(fam_final * 255), (1, 2, 0)))

    if debug:
        _save(
            output_img,
            'fam_final',
        )

    return output_img


def get_outer_quad(inner_quad, full=None, debug=False):
    inner_quad = [int(round(x)) for x in list(inner_quad)]

    buf = c.ALIGN_TRANSFORM_SIZE // 4
    aligned_outer_coords = [
        -buf, -buf, # nw
        -buf, c.ALIGN_TRANSFORM_SIZE + buf, # sw
        c.ALIGN_TRANSFORM_SIZE + buf, c.ALIGN_TRANSFORM_SIZE + buf, # se
        c.ALIGN_TRANSFORM_SIZE + buf, -buf # ne
    ]
    outer_quad = _unalign_coords(
        aligned_outer_coords, inner_quad, c.ALIGN_TRANSFORM_SIZE)

    if debug:
        full_quads = _draw_quads(full, [inner_quad, outer_quad])
        _save(full_quads, 'full_outer_quad')

    return np.array(outer_quad, dtype=np.float32)


def _mask_and_inpaint(inpainter, img, mask):
    # convert to tensor and transfer to gpu
    to_tensor = transforms.ToTensor()
    img = to_tensor(img).to('cuda').unsqueeze(0)
    mask = to_tensor(mask).to('cuda').unsqueeze(0)

    # mask and inpaint
    with torch.no_grad():
        out = img * (1. - mask) + inpainter(img, mask) * mask

    # convert to img
    out = out.squeeze().cpu().numpy()
    return out


def _reinsert(fam_og, solo_unaligned, solo_unaligned_mask):
    return Image.composite(solo_unaligned, fam_og, solo_unaligned_mask)


def _inverse_align(
    solo_aligned,
    quad,
    w, h,
    needs_boundary_mask=False,
    debug=False,
):
    solo_aligned_resized = solo_aligned.resize(
        (c.ALIGN_TRANSFORM_SIZE, c.ALIGN_TRANSFORM_SIZE),
        Image.ANTIALIAS,
    )

    wrapper_coords = (
        0, 0,  # nw
        0, h,  # sw
        w, h,  # se
        w, 0,  # ne
    )
    wrapper_coords_aligned = _align_coords(
        wrapper_coords,
        quad,
        c.ALIGN_TRANSFORM_SIZE,
    )
    wrapper_coords_aligned = [
        int(round(x)) for x in list(wrapper_coords_aligned)
    ]

    solo_aligned_padded, offset_x, offset_y = _pad(
        solo_aligned_resized,
        wrapper_coords_aligned,
    )
    wrapper_coords_aligned_padded = _shift_coords(
        wrapper_coords_aligned,
        offset_x,
        offset_y,
    )

    solo_aligned_mask = _create_rect_mask(
        solo_aligned_padded.size[0],
        solo_aligned_padded.size[1],
        (
            offset_x + c.ALIGNED_MASK_BUFFER,
            offset_y + c.ALIGNED_MASK_BUFFER,
            offset_x + c.ALIGN_TRANSFORM_SIZE - c.ALIGNED_MASK_BUFFER,
            offset_y + c.ALIGN_TRANSFORM_SIZE - c.ALIGNED_MASK_BUFFER,
        ),
    )

    if needs_boundary_mask:
        boundary_mask_aligned = _create_boundary_mask(
            solo_aligned_padded.size[0],
            solo_aligned_padded.size[1],
            (
                offset_x,
                offset_y,
                offset_x + c.ALIGN_TRANSFORM_SIZE,
                offset_y + c.ALIGN_TRANSFORM_SIZE,
            ),
        )

    if debug:
        _save(solo_aligned_padded, 'solo_aligned_padded')
        _save(solo_aligned_mask, 'solo_aligned_mask')
        if needs_boundary_mask:
            _save(boundary_mask_aligned, 'boundary_mask_aligned')

    solo_unaligned = _unalign_img(
        solo_aligned_padded,
        wrapper_coords_aligned_padded,
        w, h,
    )

    solo_unaligned_mask = _unalign_img(
        solo_aligned_mask,
        wrapper_coords_aligned_padded,
        w, h,
    )

    if needs_boundary_mask:
        boundary_mask = _unalign_img(
            boundary_mask_aligned,
            wrapper_coords_aligned_padded,
            w, h,
        )

    if debug:
        _save(solo_unaligned, 'solo_unaligned')
        _save(solo_unaligned_mask, 'solo_unaligned_mask')
        if needs_boundary_mask:
            _save(boundary_mask, 'boundary_mask')

    if needs_boundary_mask:
        return solo_unaligned, solo_unaligned_mask, boundary_mask
    return solo_unaligned, solo_unaligned_mask



def _shift_coords(coords, dx, dy):
    shifted_coords = []
    for i, val in enumerate(coords):
        if i % 2:
            shifted = coords[i] + dy
        else:
            shifted = coords[i] + dx
        shifted_coords.append(shifted)
    return shifted_coords


def _unalign_img(img, quad, w, h):
    return img.transform((w, h), Image.QUAD, quad, Image.BILINEAR)


def _pad(img, quad):
    offset_x = -min(quad[::2])
    offset_y = -min(quad[1::2])
    padded_w = max(quad[::2]) + offset_x
    padded_h = max(quad[1::2]) + offset_y
    padded_img = Image.new(img.mode, (padded_w, padded_h), (255, 255, 255))
    padded_img.paste(img, (offset_x, offset_y))
    return padded_img, offset_x, offset_y


def _unalign_coords(coords, quad, transform_size):
    nw = _unalign_point(coords[0], coords[1], quad, transform_size)
    sw = _unalign_point(coords[2], coords[3], quad, transform_size)
    se = _unalign_point(coords[4], coords[5], quad, transform_size)
    ne = _unalign_point(coords[6], coords[7], quad, transform_size)
    return nw + sw + se + ne


def _unalign_point(x, y, quad, transform_size):
    nwx, nwy, swx, swy, sex, sey, nex, ney = quad
    w = transform_size
    h = transform_size

    if USE_PIXEL_CENTER:
        x += 0.5
        y += 0.5

    x_prime = nwx + x * (nex - nwx) / w + y * (swx - nwx) / h + \
              x * y * (sex - swx - nex + nwx) / (w * h)

    y_prime = nwy + x * (ney - nwy) / w + y * (swy - nwy) / h + \
              x * y * (sey - swy - ney + nwy) / (w * h)

    return x_prime, y_prime


def _align_coords(coords, quad, transform_size):
    nw = _align_point(coords[0], coords[1], quad, transform_size)
    sw = _align_point(coords[2], coords[3], quad, transform_size)
    se = _align_point(coords[4], coords[5], quad, transform_size)
    ne = _align_point(coords[6], coords[7], quad, transform_size)
    return nw + sw + se + ne


def _align_point(x_prime, y_prime, quad, transform_size):
    w = transform_size
    h = transform_size

    # name of each coorinate in quad
    # (e.g. x coord of southwest corner = swx)
    nwx, nwy, swx, swy, sex, sey, nex, ney = quad

    # deltas between the coords
    # (e.g. change in x coord between the two north corners = dnx)
    dnx = nex - nwx
    dwx = swx - nwx
    dsx = sex - swx
    dny = ney - nwy
    dwy = swy - nwy
    dsy = sey - swy

    # define some shorthands that will be useful
    a0 = nwx
    a1 = dnx / w
    a2 = dwx / h
    a3 = (dsx - dnx) / (w * h)
    b0 = nwy
    b1 = dny / w
    b2 = dwy / h
    b3 = (dsy - dny) / (w * h)

    '''
    the problem can now be defined as:

    solve the following system of equations for x and y

    x_prime = a0 + a1 * x + a2 * y + a3 * x * y
    y_prime = b0 + b1 * x + b2 * y + b3 * x * y
    '''

    # additional shorthands for the next step
    p = b2 * a3 - b3 * a2
    q = (b0 * a3 - b3 * a0) + (b2 * a1 - b1 * a2) + (b3 * x_prime - a3 * y_prime)
    r = (b0 * a1 - b1 * a0) + (b1 * x_prime - a1 * y_prime)

    '''
    solving for y without any x terms yields this quadradic equation:

    p * y^2 + q * y + r = 0

    if p == 0, the solution is:

    y = -r / q

    otherwise, the solution is:

    y = (-q +/- sqrt(q^2 - 4 * p * r)) / (2 * p)

    after inspection, it seems +/- should always be treated as +
    (likely because of the constraints imposed via the ordering of the corners)
    '''
    if p == 0.:
        y = -r / q
    else:
        sqrt_term = math.sqrt(q ** 2 - 4 * p * r)
        y = (-q + sqrt_term) / (2 * p)
        # y = (-q - sqrt_term) / (2 * p)

    # using this solution for y, solving for x yields:
    x = (x_prime - a0 - a2 * y) / (a1 + a3 * y)

    if USE_PIXEL_CENTER:
        x -= 0.5
        y -= 0.5

    return x, y


def _draw_quad(img, quad):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.line((quad[0], quad[1], quad[2], quad[3]), fill=(255, 0, 0))
    draw.line((quad[2], quad[3], quad[4], quad[5]), fill=(255, 0, 0))
    draw.line((quad[4], quad[5], quad[6], quad[7]), fill=(255, 0, 0))
    draw.line((quad[6], quad[7], quad[0], quad[1]), fill=(255, 0, 0))
    return img


def _draw_quads(img, quads):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for i, quad in enumerate(quads):
        assert i < 3, 'TODO'
        fill = [0, 0, 0]
        fill[i] = 255
        fill = tuple(fill)
        draw.line((quad[0], quad[1], quad[2], quad[3]), fill=fill)
        draw.line((quad[2], quad[3], quad[4], quad[5]), fill=fill)
        draw.line((quad[4], quad[5], quad[6], quad[7]), fill=fill)
        draw.line((quad[6], quad[7], quad[0], quad[1]), fill=fill)
    return img


def _create_rect_mask(full_w, full_h, rect):
    mask = Image.new('1', (full_w, full_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(rect, fill=1)
    return mask


def _create_boundary_mask(full_w, full_h, rect):
    thickness = int((rect[2] - rect[0]) * BOUNDARY_MASK_REL_SIZE)
    overlap = thickness * BOUNDARY_MASK_INNER_REL_OVERLAP
    rect = (
        rect[0] - thickness + overlap,
        rect[1] - thickness + overlap,
        rect[2] + thickness - overlap,
        rect[3] + thickness - overlap,
    )

    mask = Image.new('1', (full_w, full_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(rect, outline=1, width=thickness)
    return mask


def _save(img, name):
    img.save(f'/home/asiu/data/tmp/inverse/{name}.png', 'PNG')
