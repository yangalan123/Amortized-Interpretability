#device=$1
#seed=$2
#explainer="svs-2000"
#model="bert"
#task=$3
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done
echo "device_${device}_task_${task}_model_${model}_explainer_${explainer}_seed_${seed}"
CUDA_VISIBLE_DEVICES=${device} python run_explainer.py -c configs/${task}/${model}/${explainer}.jsonnet -seed ${seed} -bsz ${batch_size}
