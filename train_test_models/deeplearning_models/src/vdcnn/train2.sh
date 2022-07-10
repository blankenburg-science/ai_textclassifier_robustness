#!/usr/bin/env bash
cd ../../


#dataset="yelp_polarity" #2 classes
#dataset="sogou_news" #5 n_classes
dataset="db_pedia" #14 classes

#dataset="yelp_review" #5 classes
#dataset="yelp_polarity" #2 classes
#dataset="amazon_review" #5 n_classes
#dataset="amazon_polarity" #2 n_classes
#dataset="sogou_news" #5 n_classes
#dataset="yahoo_answer" #10 n_classes
#dataset="ag_news" #4 classes

data_folder="datasets/${dataset}/vdcnn"
model_folder="models/vdcnn/${dataset}"
#9, 29, 49
depth=49
solver='sgd'
momentum=0.9
gamma=0.9
lr_halve_interval=15
maxlen=1024
batch_size=128
epochs=50
lr=0.0001
snapshot_interval=10
gpuid=1
nthreads=4
fold=1

for fold in 0 1 2 3 4 5 6 7 8 9
do
  python -W ignore -m src.vdcnn.train --dataset ${dataset} \
                         --model_folder ${model_folder} \
                         --data_folder ${data_folder} \
                         --depth ${depth} \
                         --solver ${solver} \
                         --maxlen ${maxlen} \
                         --batch_size ${batch_size} \
                         --epochs ${epochs} \
                         --lr ${lr} \
                         --lr_halve_interval ${lr_halve_interval} \
                         --momentum ${momentum} \
                         --gamma ${gamma} \
                         --snapshot_interval ${snapshot_interval} \
                         --gpuid ${gpuid} \
                         --nthreads ${nthreads} \
                         --shortcut \
                         --fold ${fold}
done
