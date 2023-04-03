#FROM winnietsang/onnx-tensorflow:onnx1.7.0-tf2.2
FROM tensorflow/tensorflow:latest-devel
# onnx
#WORKDIR /tmp
#RUN git clone https://github.com/onnx/onnx.git
#WORKDIR /tmp/onnx
#RUN git submodule update --init --recursive
#RUN python3 setup.py install
RUN apt-get update
RUN apt-get install -y protobuf-compiler libprotoc-dev
#RUN conda install -c conda-forge protobuf numpy
#RUN pip install onnx
# onnx-tf
WORKDIR /tmp
RUN git clone https://github.com/onnx/onnx-tensorflow.git
WORKDIR /tmp/onnx-tensorflow
RUN pip install -e .
#RUN pip install onnx-tf
# user
RUN useradd -m asiu
RUN groupadd asig
RUN usermod -a -G asig asiu
USER asiu
ENV PYTHONPATH "${PYTHONPATH}:/home/asiu/code"
ENV PATH "${PATH}:/home/asiu/.local/bin"
ENV ASI_CODE "/home/asiu/code/asi"
ENV ASI_DATASETS "/home/asiu/datasets"
ENV ASI_DATA "/home/asiu/data"
WORKDIR /home/asiu
RUN pip install onnx
RUN pip install tensorflowjs
RUN pip install tensorflow-addons
