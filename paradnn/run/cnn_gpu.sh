data_type="float16"
platform="gpu"
cnnblock='bottleneck'
outpath=../output/${cnnblock}_${platform}_${data_type}
mkdir -p $outpath


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
for input_size in 100 #200 300
do
for output_size in 500 #1000 1500
do 
for bs in 64 128 # 256 512 1024
do

name=block_${nblock}-filtersz_${filters}-input_${input_size}-output_${output_size}-batchsize_${bs}

# skip the experiment if its performance report exists
grep "examples/sec" $outpath/$name.err > tmp
filesize=$(stat -c%s tmp)
if [ "${filesize}" -gt 0 ];
then
  echo "skipping "$name
  continue
fi

echo "running "$name
python cnn.py --use_tpu=${use_tpu} --tpu_name=${tpu_name} --data_type=${data_type} \
              --block_fn=${cnnblock} --filters=${filters} \
              --resnet_layers=${nblock},${nblock},${nblock},${nblock}\
              --input_size=${input_size} --output_size=${output_size}\
              --batch_size=${bs} --train_steps=300 \
              1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
done
done
