#!/bin/sh
# default setting
epoch=30
#train_bsz=10
# for fastshap-only
train_bsz=1
explainer="svs"
# will be overridden by named arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done
for lr in 5e-5
do
for seed in ${seed}
do
  CUDA_VISIBLE_DEVICES=${device} python run.py --seed ${seed}  --lr ${lr} -e ${epoch} --train_bsz ${train_bsz} --explainer ${explainer} --topk 10 --task ${task} -tm ${target_model} --storage_root ${output_dir}
  #python run.py --discrete --seed ${seed} --lr ${lr} -e ${epoch} --train_bsz ${train_bsz} --explainer ${explainer}
  #python run.py --seed ${seed} --lr ${lr} -e ${epoch} --train_bsz ${train_bsz} --explainer ${explainer}
done
done
