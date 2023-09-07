FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
#FROM python:3.8-slim

# use an older system (18.04) to avoid opencv incompatibility (issue#3524)
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
#ARG USER_ID=1000
#RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER appuser
#WORKDIR /home/appuser


#ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard cmake   # cmake from apt-get is too old
# install detectron2
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install pytorch
# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
#RUN git clone https://github.com/microsoft/unilm.git
#COPY requirements.txt requirements.txt
#RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache

# CPU only
#RUN pip install torch==1.9.1+cpu torchvision==0.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app
COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/supervisor/conf.d/

# install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip3 install -r requirements.txt 

RUN wget https://layoutlm.blob.core.windows.net/dit/dit-fts/funsd_dit-l_mrcnn.pth
COPY detection/. /app/
#COPY /cofigs/ /app/cofigs/
#COPY /ditod/ /app/ditod/


ENV checkpoint_file=/app/funsd_dit-l_mrcnn.pth
ENV config_file=/app/configs/mask_rcnn_dit_large.yaml

#CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

COPY . ./

EXPOSE $PORT

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 _wsgi:app
