data_type="float16"
platform="gpu"
rnncell='rnn'
outpath=../output/${rnncell}_${platform}_${data_type}
mkdir -p $outpath


use_tpu="False"
tpu_name=$HOSTNAME
if [ "${platform}" = "tpu" ];
then
  use_tpu="True"
fi


for layer in 1 #5 9 13
do
for maxlength in 10 #50 90
do
for vocabsize in 1 #5 9
do 
for embeddingsize in 100 #500 900
do
for bs in 64 #128 256 512 1024
do

name=layer_${layer}-maxlength_${maxlength}-vocabsize_${vocabsize}-embeddingsize_${embeddingsize}-batchsize_${bs}

# skip the experiment if its performance report exists
grep "examples/sec" $outpath/$name.err > tmp
filesize=$(stat -c%s tmp)
if [ "${filesize}" -gt 0 ];
then
  echo "skipping "$name
  continue
fi

echo "running "$name
python rnn.py --use_tpu=${use_tpu} --tpu_name=${tpu_name} \
              --input_size=${maxlength},${vocabsize},${embeddingsize} --layer=${layer} \
              --batch_size=${bs} --train_steps=300 \
              1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
done
done
