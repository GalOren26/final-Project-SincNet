FROM pytorch/pytorch

RUN apt-get -y update && \
    apt-get -y install vim \
                       htop \
                       git \
                       wget \
                       sudo \
                       software-properties-common \
                       unzip \
                       tmux \
                       tree \
                       bash-completion \
                       python3-pip   

COPY requirements.txt /root/requirements.txt
RUN . ~/.bashrc && pip install -r /root/requirements.txt

RUN mkdir /home/sincnet

ADD . /home/sincnet

WORKDIR /home/sincnet

ENV PYTHONPATH "${PYTHONPATH}:/home/sincnet/src"
