#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import clip
from PIL import Image
import pickle
import copy


def get_align(out,dt,model,preprocess):
    imgs=out
    imgs1=imgs.reshape([-1]+list(imgs.shape[2:]))

    tmp=[]
    for i in range(len(imgs1)):

        img=Image.fromarray(imgs1[i])
        image = preprocess(img).unsqueeze(0).to(device)
        tmp.append(image)

    image=torch.cat(tmp)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    image_features1=image_features.cpu().numpy()

    image_features1=image_features1.reshape(list(imgs.shape[:2])+[512])

    fd=image_features1[:,1:,:]-image_features1[:,:-1,:]

    fd1=fd.reshape([-1,512])
    fd2=fd1/np.linalg.norm(fd1,axis=1)[:,None]

    tmp=np.dot(fd2,dt)
    m=tmp.mean()
    acc=np.sum(tmp>0)/len(tmp)
    print(m,acc)
    return m,acc


def split_s(ds_p,M,if_std):
    all_ds=[]
    start=0
    for i in M.mindexs:
        tmp=M.stylespace[i].shape[1]
        end=start+tmp
        tmp=ds_p[start:end]
#        tmp=tmp*M.code_std[i]

        all_ds.append(tmp)
        start=end

    all_ds2=[]
    tmp_index=0
    for i in range(len(M.s_names)):
        if (not 'RGB' in M.s_names[i]) and (not len(all_ds[tmp_index])==0):

#            tmp=np.abs(all_ds[tmp_index]/M.code_std[i])
#            print(i,tmp.mean())
#            tmp=np.dot(M.latent_codes[i],all_ds[tmp_index])
#            print(tmp)
            if if_std:
                tmp=all_ds[tmp_index]*M.code_std[i]
            else:
                tmp=all_ds[tmp_index]

            all_ds2.append(tmp)
            tmp_index+=1
        else:
            tmp=np.zeros(len(M.stylespace[i][0]))
            all_ds2.append(tmp)
    return all_ds2


imagenet_templates = [
    'a bad photo of a {}.',
#    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def zeroshot_classifier(classnames, templates,model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def get_dt(classnames,model):
    text_features=zeroshot_classifier(classnames, imagenet_templates,model).t()

    dt=text_features[0]-text_features[1]
    dt=dt.cpu().numpy()

#    t_m1=t_m/np.linalg.norm(t_m)
#    dt=text_features.cpu().numpy()[0]-t_m1
    print(np.linalg.norm(dt))
    dt=dt/np.linalg.norm(dt)
    return dt


def get_boundary(fs3,dt,M,threshold):
    tmp=np.dot(fs3,dt)

    ds_imp=copy.copy(tmp)
    select=np.abs(tmp)<threshold
    num_c=np.sum(~select)


    ds_imp[select]=0
    tmp=np.abs(ds_imp).max()
    ds_imp/=tmp

    boundary_tmp2=split_s(ds_imp,M,if_std=True)
    print('num of channels being manipulated:',num_c)
    return boundary_tmp2,num_c


def get_fs(file_path):
    fs=np.load(file_path+'single_channel.npy')
    tmp=np.linalg.norm(fs,axis=-1)
    fs1=fs/tmp[:,:,:,None]
    fs2=fs1[:,:,1,:]-fs1[:,:,0,:]  # 5*sigma - (-5)* sigma
    fs3=fs2/np.linalg.norm(fs2,axis=-1)[:,:,None]
    fs3=fs3.mean(axis=1)
    fs3=fs3/np.linalg.norm(fs3,axis=-1)[:,None]
    return fs3
