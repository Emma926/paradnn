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

2. For example, to run FC models
```
cd paradnn/
bash run/fc_cpu.sh
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
