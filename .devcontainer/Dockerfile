
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04





ENV DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    wget\
    manpages-dev\
    g++\
    gcc\
    nodejs\
    libssl-dev\
    unzip\
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
    && rm -rf /var/lib/apt/lists/*
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda


RUN apt update
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9
RUN apt install -y python3-pip
# RUN apt-get -y purge python3.8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN update-alternatives --set python /usr/bin/python3.9

RUN apt-get install python3-dev 



RUN python3.9 -m pip install --upgrade pip

# set up user
ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR ${HOME}


RUN python3.9 -m pip install --no-cache-dir jupyter matplotlib
# RUN    pip install --no-cache-dir "intensity-normalization[ants]"
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
RUN python3.9 -m  pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
RUN python3.9 -m  pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html



#RUN    pip install pyradiomics
RUN python3.9 -m  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

RUN python3.9 -m  pip install \
    #itk \
    pytorch-ignite==0.4.8 \
    PyWavelets==1.3.0 \
    scipy==1.8.1 \
    nibabel==3.2.2 \
    pillow!=8.3.0 \  
    tensorboard==2.9.0 \
    scikit-image==0.19.2 \
    tqdm>=4.47.0 \
    lmdb==1.3.0 \
    flake8>=3.8.1 \
    flake8-bugbear \
    flake8-comprehensions \
    flake8-executable \
    flake8-pyi \
    pylint!=2.13 \
    mccabe \
    pep8-naming \
    pycodestyle \
    pyflakes==2.4.0 \
    black \
    isort \
    pytype>=2020.6.1 \
    types-pkg_resources \
    mypy>=0.790 \
    ninja==1.10.2.3 \
    #torchvision==0.12.0 \
    opencv-python==4.5.5.64 \
    psutil==5.9.1 \
    Sphinx==3.5.3 \
    recommonmark==0.6.0 \
    sphinx-autodoc-typehints==1.11.1 \
    sphinx-rtd-theme==0.5.2 \
    # cucim==22.2.1; platform_system == "Linux" \
    # imagecodecs; platform_system == "Linux" \
    # tifffile; platform_system == "Linux" \
    pandas==1.4.2 \
    requests==2.27.1 \
    einops==0.4.1 \
    transformers==4.19.2 \
    mlflow==1.26.1 \
    matplotlib!=3.5.0 \
    #tensorboardX==2.5 \
    types-PyYAML \
    pyyaml==6.0 \
    fire==0.4.0 \
    jsonschema==4.6.0 \
    kymatio==0.2.1 \
    juliacall==0.8.0 \
    jupyterlab_github==3.0.1 \
    torchio==0.18.77 \
    pytorch-lightning==1.6.4 \
    git+https://github.com/sat28/githubcommit.git \
    gdown==4.4.0 \
    seaborn==0.11.2 \
    # PyTorchUtils==0.0.3 \
    # optuna==2.10.0 \
    jupyterlab-git==0.37.1 \
    comet-ml==3.31.5 \
    # voxelmorph==0.2 \
    # tensorflow==2.9.1 \
    # tensorflow_addons==0.17.0 \
    ipywidgets==7.7.0 \
    h5py==3.7.0 \
    itk-elastix==0.14.1 \
    zenodopy==0.2.0 \
    evalutils==0.3.0 \
    nnunet==1.7.0 \
    SimpleITK \
    git+https://github.com/balbasty/nitorch#egg=nitorch[all] \
    threadpoolctl==3.1.0 \
    batchgenerators==0.24 \
    # zenodo_get==1.3.4 \
    git+https://github.com/DIAGNijmegen/picai_baseline \
    git+https://github.com/DIAGNijmegen/picai_prep \
    git+https://github.com/DIAGNijmegen/picai_eval \
    #https://github.com/DIAGNijmegen/picai_prep/archive/refs/tags/v1.3.2.zip \
    KevinSR==0.1.19 \
    dask==2022.6.0 \
    intensity-normalization[ants] \
    # numba==0.55.2 \
    # pyro-ppl==1.8.1 \
    # warp-lang==0.2.3 \
    report_guided_annotation \
    SimpleITK-SimpleElastix \
    geomloss[full] \
    torch-optimizer \ 
    optax \ 
    ott-jax \ 
    evojax[extra] \
    jraph \
    flax




# ENV JAXLIB_VERSION=0.3.16
RUN apt-get install python-dev -y
RUN apt-get install python3.9-dev -y

RUN pip install pycocotools
RUN python3.9 -m  pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'
# RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \


RUN git clone https://github.com/google-research/scenic.git ${HOME}/build
RUN cd ${HOME}/build/scenic
RUN python3.9 -m  pip install ${HOME}/build
RUN python3.9 -m pip install --upgrade jax jaxlib


#olld one
RUN git clone https://github.com/MIC-DKFZ/PROUNET.git ${HOME}/data/baseline/old
#from picai
RUN git clone https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation.git ${HOME}/data/baseline/mrusProstate
RUN git clone https://github.com/DIAGNijmegen/picai_nnunet_semi_supervised_gc_algorithm.git ${HOME}/data/baseline/nnunetSemiSup
RUN git clone https://github.com/DIAGNijmegen/prostateMR_3D-CAD-csPCa.git ${HOME}/data/baseline/#Dcad
RUN git clone https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm.git ${HOME}/data/baseline/seemiSupUnet



# RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# RUN add-apt-repository \
# "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
# $(lsb_release -cs) stable"
# RUN apt-get update
# RUN apt-get install docker-ce docker-ce-cli containerd.io

# RUN curl https://get.docker.com | sh \
#   && sudo systemctl --now enable docker

# RUN apt-get update
# RUN apt-get install -y nvidia-docker2
# # RUN systemctl restart docker

# RUN docker run -v /var/run/docker.sock:/var/run/docker.sock \
#     docker run --rm --gpus all \
#     nvidia/cuda:11.0-base \
#     nvidia-smi


RUN apt-get update && apt-get install -y \
        apt-utils \
        ca-certificates \
        openssh-client \
        curl \
        iptables \
        git \
        gnupg \
        supervisor && \
    rm -rf /var/lib/apt/list/*

# NVIDIA Container Toolkit & Docker
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && apt-get install -y nvidia-docker2 docker.io docker-compose && \
    rm -rf /var/lib/apt/list/*

COPY modprobe startup.sh /usr/local/bin/
COPY supervisor/ /etc/supervisor/conf.d/
COPY logger.sh /opt/bash-utils/logger.sh

RUN chmod +x /usr/local/bin/startup.sh /usr/local/bin/modprobe
VOLUME /var/lib/docker



RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l

ENTRYPOINT ["startup.sh"]
CMD ["sh"]