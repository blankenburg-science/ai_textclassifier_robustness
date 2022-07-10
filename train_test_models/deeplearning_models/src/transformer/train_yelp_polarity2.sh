#!/usr/bin/env bash
cd ../../

# base model (GPU ram > 8GB): embedding_dim=512, attention_dim=64, n_heads=8, n_layers=6, dropout=0.1, n_warmup_step=4000, batch_size=64
# big model (GPU ram > 24GB): embedding_dim=1024, attention_dim=64, n_heads=16, n_layers=6, dropout=0.1, n_warmup_step=4000, batch_size=64
# beware when max_sequence_length=-1, it will pad to the longest sequence which can be very long and cause GPU memory error

dataset="yelp_polarity" #2 classes

data_folder="datasets/${dataset}/transformer"
model_folder="models/transformer/${dataset}"
attention_dim=1024 #128,512,1024
n_heads=16 #4,8,16
n_layers=6 #4,6,6
maxlen=250 # longest sequence
dropout=0.1
n_warmup_step=4000
batch_size=56 #64
epochs=40
snapshot_interval=5
gpuid=1
nthreads=6
use_all_gpu=0
wait_minutes=5

for fold in 3
do
  python -m src.transformer.train --dataset ${dataset} \
                                   --data_folder ${data_folder} \
                                   --model_folder ${model_folder} \
                                   --attention_dim ${attention_dim} \
                                   --n_heads ${n_heads} \
                                   --n_layers ${n_layers} \
                                   --maxlen ${maxlen} \
                                   --dropout ${dropout} \
                                   --n_warmup_step ${n_warmup_step} \
                                   --batch_size ${batch_size} \
                                   --epochs ${epochs} \
                                   --snapshot_interval ${snapshot_interval} \
                                   --gpuid ${gpuid} \
                                   --nthreads ${nthreads} \
                                   --use_all_gpu ${use_all_gpu} \
                                   --fold ${fold} \
                                   --wait_minutes ${wait_minutes}
done
