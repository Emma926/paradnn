import os
import math
import numpy as np
import json
import sys

root = '../output/'

if len(sys.argv) > 1:
  path = root + sys.argv[1]
  if not os.path.isdir(path):
    print('Error: ', path, ' is not a directory.')
    choices = os.listdir(root)
    if len(choices) == 1:
      print('There is only one directory. Running', choices[0])
      path = root + choices[0]
    else:
      print('Available directories: ', os.listdir(root))
      exit()
else:
  choices = os.listdir(root)
  if len(choices) == 1:
    print('There is only one directory. Running', choices[0])
    path = root + choices[0]
  else:
    print('Please the input directory.')
    print('Available directories: ', os.listdir(root))
    exit()
  

files = os.listdir(path)
label_all = []
param_all = []
intensity_all = []
flops_all = []
step_per_sec_all = []
example_per_sec = []
fl_all = []
err_jobs = []
states = []
versions = set()

for f in files:
  if not '.out' in f:
    continue
  out_f = os.path.join(path, f)
  out_fin = open(out_f,'r')

  state = ''
  param = -1
  flops = -1
  version = ''
  for line in out_fin:
    if 'Tensorflow version' in line or 'Tensoflow version' in line:
      version = line.strip('\n').split(' ')[-1]
      versions.add(version)
    elif '_TFProfRoot' in line and 'params' in line:
      line = line.split('/')[-1].split(' ')[0]
      if line[-1] == 'k':
        param = float(line.strip('k')) * 1e3
      if line[-1] == 'm':
        param = float(line.strip('m')) * 1e6
      if line[-1] == 'b':
        param = float(line.strip('b')) * 1e9
    elif '_TFProfRoot' in line and 'flops' in line:
      line = line.split('/')[-1].split(' ')[0]
      if line[-1] == 'k':
        flops = float(line.strip('k')) * 1e3
      if line[-1] == 'm':
        flops = float(line.strip('m')) * 1e6
      if line[-1] == 'b':
        flops = float(line.strip('b')) * 1e9
  if flops == -1 or param == -1:
    state = 'not run'
  err_f = os.path.join(path, f.replace('.out', '.err'))
  err_fin = open(err_f, 'r')
  speed = []
  flag = 0
  for line in err_fin:
    if "global_step/sec" in line:
      speed.append(float(line.strip('.\n').split(' ')[-1]))
    if 'final step' in line:
      flag = 1
    # check the states
    if 'OOM' in line:
      state = 'OOM'
    elif 'Socket closed' in line:
      state = 'Socket closed'
    elif 'Deadline Exceeded' in line:
      state = 'Deadline Exceeded'
    elif 'Negative dimension size' in line:
      state = 'Negative dimension size' 
    elif 'NaN value' in line:
      state = 'NaN value' 
    elif 'unhealthy' in line:
      state = 'unhealthy' 
    elif 'FAILED state' in line:
      state = 'FAILED state' 
  if speed == [] and flag == 1:
    state = 'finished but no performance'
    err_jobs.append(f[:-4])
    states.append(state)
    continue
  if speed == [] and state == '':
    state = 'no error but not finished'
    err_jobs.append(f[:-4])
    states.append(state)
    continue
  if state != '':
    err_jobs.append(f[:-4])
    states.append(state)
    continue

  step_per_sec = max(speed)

  # get batch size
  bs = 128
  if 'bs' in f or 'batchsize' in f:
    for tmp in f.replace('.out','').split('-'):
      if 'bs' in tmp or 'batchsize' in tmp:
        bs = int(tmp.split('_')[-1])

  flops_per_sec = flops * step_per_sec / 1e9
  if math.isnan(flops_per_sec):
    continue
  example_per_sec.append(bs*step_per_sec) 
  flops_all.append(flops_per_sec)
  param_all.append(param)
  if "_b16" in path:
    intensity_all.append(flops/(param*2.0))
  else:
    intensity_all.append(flops/(param*4.0))
  label_all.append(f[:-4])
  step_per_sec_all.append(step_per_sec)
  fl_all.append(flops / 1e9)

d = {
  'labels': label_all,
  'intensity': intensity_all,
  'flops': flops_all,
  'params': param_all,
  'step_per_sec': step_per_sec_all,
  'fl': fl_all,
  'example_per_sec': example_per_sec,
  'err_jobs': err_jobs,
  'states': states,
}

print('error states:', set(states))

name_map = {
  'fc': 'fc',
  'fc_free_input': 'fc_free_data',
  'conv': 'cnn_vgg',
  'conv_free_data': 'cnn_vgg_free_data',
  'conv_resnetlike': 'cnn_resnet',
  'conv_resnetlike_free_data': 'cnn_resnet_free_data',
  'lstm_small': 'lstm'
}

name = ''

if path.split('/')[-1] in name_map:
  name += name_map[path.split('/')[-1]]
else:
  name += path.split('/')[-1]

print('TensorFlow versions:', versions)

if os.path.isfile('data/' + name + '.json'):
  with open('data/' + name + '.json','r') as infile:
    din = json.load(infile)
  print(len(label_all) - len(din['labels']), 'more data points than last time.')

if not os.path.isdir('data'):
    os.makedirs('data')

with open('data/' + name + '.json','w') as outfile:
    json.dump(d, outfile)
    print('Results written in data/' + name + '.json')

print('Total data points: ', len(label_all))
print('Total files: ', len(files)/2)
print('Missing: ', len(files)/2 - len(label_all))
print('Error files: ', len(err_jobs))
