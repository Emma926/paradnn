# paradnn (alpha)


![ParaDnn](paradnn.pdf)

ParaDnn is a tool that generates parameterized deep neural network models.
It provides large “end-to-end” models covering current and future applications, and parameterizing the models to explore a much larger design space of DNN model attributes.
If you use paradnn for your project, please cite our paper:
```
Wang, Yu Emma, and Gu-Yeon Wei and David Brooks. "Benchmarking TPU, GPU, and CPU Platforms for Deep Learning." arXiv preprint arXiv:1907.10701 (2019).
```
```
@article{wang2019benchmarking,
  title={Benchmarking TPU, GPU, and CPU Platforms for Deep Learning},
  author={Wang, Yu Emma and Wei, Gu-Yeon and Brooks, David},
  journal={arXiv preprint arXiv:1907.10701},
  year={2019}
}
```


This repository also includes the analysis tools demonstrated in the paper.


1. To test
```
python test.py --use_tpu $USE_TPU
```

2.1 For example, to run FC models on CPUs, first modify the hyperparameter ranges in
`paradnn/run/fc_cpu.sh`, and do
```
cd paradnn/
bash run/fc_cpu.sh
```

2.2 To run FC models on TPUs, first modify the hyperparameter ranges in file
`paradnn/run/fc_tpu.sh`, and do
```
cd paradnn/
bash run/fc_tpu.sh
```

2.3 To run FC models on TPUs, first modify the hyperparameter ranges and `gcp_bucket` in file
`paradnn/run/fc_tpu_trace.sh`, and do
```
cd paradnn/
bash run/fc_tpu_trace.sh
```

To download the data from the traces
```
cd ../scripts
bash download_jsonfiles_parallel.py $trace_folder_name $gcp_bucket 
```

To parse the downloaded data
```
bash parse_jsonfiles.py $trace_folder_name 
```

3. To collect data
```
cd ../scripts
python get_perf.py
```

4. To run the analysis tools
```
cd scripts/plotting
jupyter notebook
```




Yu (Emma) Wang  
9/13/2019
