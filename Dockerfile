FROM nvcr.io/nvidia/pytorch:20.12-py3

# idk man
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# python deps
RUN pip install scipy
RUN pip install scikit-image
RUN pip install opencv-python flask dlib google-cloud-storage pandas lmdb
RUN pip install lpips
RUN pip install kornia
RUN pip install adabelief-pytorch
RUN pip install ranger-adabelief
RUN pip install marshmallow
RUN pip install gcsfs
RUN pip install google-cloud-logging
RUN pip install google-cloud-secret-manager
RUN pip install pyspng
RUN pip install azure-cognitiveservices-vision-face
RUN pip install torchdiffeq
RUN pip install git+https://github.com/calvinpelletier/CLIP.git
RUN pip install wandb

# tfjs conversion
# RUN pip install tensorflow
# WORKDIR /tmp
# RUN git clone https://github.com/onnx/onnx-tensorflow.git
# WORKDIR /tmp/onnx-tensorflow
# RUN pip install -e .
# RUN pip install onnx
# RUN pip install tensorflowjs
# RUN pip install tensorflow-addons

# other deps
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-opencv

# apache tvm
RUN apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
WORKDIR /tmp
RUN git clone --recursive https://github.com/apache/tvm tvm
WORKDIR /tmp/tvm
RUN mkdir build
RUN cp cmake/config.cmake build


# user
RUN useradd -m asiu
RUN groupadd asig
RUN usermod -a -G asig asiu
USER asiu
ENV PYTHONPATH "${PYTHONPATH}:/home/asiu/code"
ENV ASI_CODE "/home/asiu/code/asi"
ENV ASI_DATASETS "/home/asiu/datasets"
ENV ASI_DATA "/home/asiu/data"
ENV ASI_CONFIG_FILENAME "dev_local.yaml"
ENV GOOGLE_APPLICATION_CREDENTIALS "/home/asiu/google_application_creds.json"
ENV ASI_ENV "dev"
WORKDIR /home/asiu

# idk man 2 electric boogaloo
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /workspace/entry.sh) && chmod a+x /workspace/entry.sh
ENTRYPOINT ["/workspace/entry.sh"]
