FROM nvcr.io/nvidia/tensorflow:22.01-tf1-py3

ENV DEBIAN_FRONTEND noninteractive
ADD . /code
WORKDIR /code

RUN apt-get update -y
RUN apt install -y wget

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

WORKDIR /code/deeplab/models-master/research/deeplab

ENV PYTHONPATH=$PYTHONPATH:"/code/deeplab/models-master/research"
ENV PYTHONPATH=$PYTHONPATH:"/code/deeplab/models-master/research/slim"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# To generate the dataset
#CMD python3.8 ./datasets/build_emid_data.py
# To run experiment
CMD ./emid_xception.sh
