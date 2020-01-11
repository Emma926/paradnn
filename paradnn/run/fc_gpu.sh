data_type="float32"
platform="gpu"
outpath=../output/fc_${platform}_${data_type}
mkdir -p $outpath


use_tpu="False"
if [ "${platform}" = "tpu" ];
then
  use_tpu="True"
fi

for node in 32 64 # 128 256 512 1024 2048 4096 8192
do
for layer in 4 # 8 16 32 64 128
do
for bs in 64 #128 # 256 512 1024 2048 4096 8192 16384
do

name=node_${node}-layer_${layer}-batchsize_${bs}
echo $name
python fc.py --use_tpu=${use_tpu} --data_type=${data_type} --layer=${layer} \
             --node=${node} --input_size=${node} --output_size=${node} \
              --batch_size=${bs} --train_steps=100 \
              1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
