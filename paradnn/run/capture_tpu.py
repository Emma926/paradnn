# https://cloud.google.com/tpu/docs/cloud-tpu-tools
# Install capture_tpu_profile
#(vm)$ pip freeze | grep cloud-tpu-profiler
#(vm)$ sudo pip install --upgrade "cloud-tpu-profiler==1.5.2"


import os
import sys
import time

cmd = 'capture_tpu_profile --tpu=' + os.uname()[1] + ' --duration_ms=3000 --logdir='

folder = sys.argv[1]

count = 0

print("Removing previous traces..")
os.system('gsutil rm -r ' + folder)
time.sleep(30)

while count <= 5:
  os.system(cmd + folder) 
  count += 1
  time.sleep(10)
 
