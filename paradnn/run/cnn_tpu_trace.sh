data_type="bfloat16"
platform="tpu"
cnnblock='bottleneck'
gcp_bucket="tpubenchmarking"
trace_dir=gs://${gcp_bucket}/${cnnblock}_${data_type}
tmp_dir=gs://${gcp_bucket}/tmp/ 


use_tpu="False"
tpu_name=$HOSTNAME
if [ "${platform}" = "tpu" ];
then
  use_tpu="True"
fi


input_size=200
output_size=1000

for filters in 16 #32 64
do
for nblock in 1 #2 3 4 5 6 7 8
do
for bs in 64 128 # 256 512 1024
do

name=block_${nblock}-filters_${filters}-batchsize_${bs}

echo "running "$name
python cnn.py --use_tpu=${use_tpu} --tpu_name=${tpu_name} --data_type=${data_type} \
              --block_fn=${cnnblock} --filters=${filters} \
              --resnet_layers=${nblock},${nblock},${nblock},${nblock}\
              --input_size=${input_size} --output_size=${output_size}\
              --batch_size=${bs} --train_steps=10000000 --model_dir=${tmp_dir} &

wlpid=$!
python run/capture_tpu.py ${trace_dir}
kill $wlpid

done
done
done
