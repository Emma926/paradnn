data_type="float32"
platform="gpu"
outpath=../output/fc_${platform}_${data_type}
mkdir -p $outpath


use_tpu="False"
if [ "${platform}" = "tpu" ];
then
  use_tpu="True"
fi

for layer in 4 
do
for node in 32 64 # 128 256 512 1024 2048 4096 8192
do
for input in 2000 
do
for output in 1000 
do
for bs in 64 128 # 256 512 1024 2048 4096 8192 16384
do

name=layer_${layer}-node_${node}-input_${input}-output_${output}-bs_${bs}
echo $name
python fc.py --use_tpu=${use_tpu} --data_type=${data_type} --layer=${layer} \
             --node=${node} --input_size=${node} --output_size=${node} \
              --batch_size=${bs} --train_steps=100 \
              1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
done
done
