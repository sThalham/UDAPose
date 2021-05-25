#!/bin/bash

sudo docker build --no-cache -t pyrapose_gpu_1 .
thispid=$(sudo docker run --gpus all --network=host --name=pyrapose_gpu_1 -t -d -v ~/data/train_data:/UDAPose/data -v ~/data/Meshes:/UDAPose/Meshes pyrapose_gpu_1)

#sudo nvidia-docker exec -it $thispid bash

#sudo nvidia-docker container kill $thispid
#sudo nvidia-docker container rm $thispid


