data_type="bfloat16"
platform="tpu"
# Add your GCP bucket
gcp_bucket=""
trace_dir=gs://${gcp_bucket}/fc_${data_type}
tmp_dir=gs://${gcp_bucket}/tmp/ 


use_tpu="False"
tpu_name=$HOSTNAME
if [ "${platform}" = "tpu" ];
then
  use_tpu="True"
fi

for node in 4096 #64 128 256 512 1024 2048 4096 8192
do
for layer in 8 # 8 16 32 64 128
do
for bs in 8192 #128 256 512 1024 2048 4096 8192 16384
do

name=node_${node}-layer_${layer}-batchsize_${bs}

gsutil rm -r $tmp_dir

echo "running "$name
python fc.py --use_tpu=${use_tpu} --tpu_name=${tpu_name} --data_type=${data_type} \
              --layer=${layer} --node=${node} --input_size=${node} --output_size=${node} \
              --batch_size=${bs} --train_steps=10000000 --model_dir=${tmp_dir} &

wlpid=$!
python run/capture_tpu.py ${trace_dir}
kill $wlpid

done
done
done
