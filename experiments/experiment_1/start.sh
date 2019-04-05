#!/usr/bin/env bash
chmod +x runner.sh

docker ps -a | awk '{ print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13}' | grep deep-logistics | awk '{print $1 }' | xargs -I {} docker rm -f {}

#nvidia-docker run -d --name deep-logistics --volume deep-logistics perara/deep-logistics:latest
nvidia-docker run \
-d \
--name deep-logistics \
-v deep_logistics:/data \
-v /raid/home/perara12/git/deep_logistics_ml:/root/deep_logistics_ml \
-v /raid/home/perara12/git/deep_logistics:/root/deep_logistics \
-v /raid/home/perara12/volumes/root:/root \
-p 6010:6006 \
--shm-size=64g \
--ulimit memlock=-1 \
perara/deep-logistics:latest
#-e NVIDIA_VISIBLE_DEVICES=0,1 \

#

sleep 2
docker logs --follow deep-logistics
