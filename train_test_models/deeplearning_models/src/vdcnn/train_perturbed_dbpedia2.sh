#!/usr/bin/env bash
cd ../../


#dataset="yelp_polarity" #2 classes
#dataset="ag_news" #4 classes
dataset="db_pedia" #14 classes

#dataset="yelp_review" #5 classes
#dataset="yelp_polarity" #2 classes
#dataset="amazon_review" #5 n_classes
#dataset="amazon_polarity" #2 n_classes
#dataset="sogou_news" #5 n_classes
#dataset="yahoo_answer" #10 n_classes
#dataset="ag_news" #4 classes

depth=9
solver='sgd'
momentum=0.9
gamma=0.9
lr_halve_interval=15
maxlen=1024
batch_size=64 #128
epochs=25
lr=0.0001
snapshot_interval=5
gpuid=1
nthreads=4
wait_minutes=2


for fold in {0..4}
do
  data_folder="datasets/${dataset}/vdcnn_perturb_keyboard"
  model_folder="models/vdcnn_perturb_keyboard/${dataset}"
  input_folder="perturbed/NeighborKeyboard"
  python -W ignore -m src.vdcnn.train_perturb --dataset ${dataset} \
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
                         --fold ${fold} \
                         --input_folder ${input_folder} \
                         --wait_minutes ${wait_minutes}

  data_folder="datasets/${dataset}/vdcnn_perturb_ocr"
  model_folder="models/vdcnn_perturb_ocr/${dataset}"
  input_folder="perturbed/SimilarSymbols"
  python -W ignore -m src.vdcnn.train_perturb --dataset ${dataset} \
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
                         --fold ${fold} \
                         --input_folder ${input_folder} \
                         --wait_minutes ${wait_minutes}

  data_folder="datasets/${dataset}/vdcnn_perturb_mic"
  model_folder="models/vdcnn_perturb_mic/${dataset}"
  input_folder="perturbed/HomoPhones"
  python -W ignore -m src.vdcnn.train_perturb --dataset ${dataset} \
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
                         --fold ${fold} \
                         --input_folder ${input_folder} \
                         --wait_minutes ${wait_minutes}
done
