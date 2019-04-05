#!/bin/bash
nvidia-docker rm --force per-python
nvidia-docker run \
-itd \
-p 6010:6006 \
-p 6011:22 \
--name per-python \
--shm-size=64g \
--ulimit memlock=-1 \
nvcr.io/nvidia/tensorflow:19.01-py3

nvidia-docker exec -it per-python sh -c "mkdir -p /home/per && useradd -m -d /home/per -s /bin/sh per && chown -R per:per /home/per/"
nvidia-docker exec -d per-python sh -c "echo 'per:per' | chpasswd"
nvidia-docker exec -d per-python sh -c "mkdir -p /tmp/ray && chown per:per /tmp/ray"
nvidia-docker exec -d per-python sh -c "mkdir -p /home/per/ray_results && chown per:per /home/per/ray_results"
nvidia-docker exec -d per-python sh -c "apt update && apt install openssh-server libsm6 libxext6 libxrender-dev -y && ssh-keygen -A && mkdir -p /run/sshd && /usr/sbin/sshd"
nvidia-docker exec -d per-python sh -c "tensorboard --logdir=/home/per/ray_results && tail -f /etc/hosts"
