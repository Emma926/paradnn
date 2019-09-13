import os
import subprocess
import time
import sys

root = os.path.realpath('..')
if len(sys.argv) > 1:
  exp = sys.argv[1]
  choices = os.listdir(os.path.join(root, 'generated'))
  if not os.path.isdir(os.path.join(root, 'generated', exp)) and len(choices) == 1:
    print('There is only one directory. Running', choices[0])
    exp = choices[0]
  elif not os.path.isdir(os.path.join(root, 'generated', exp)):
    print('Available directories:', os.listdir(os.path.join(root, 'generated')))
    exit()
else:
  choices = os.listdir(os.path.join(root, 'generated'))
  if len(choices) == 1:
    print('There is only one directory. Running', choices[0])
    exp = choices[0]
  else:
    print('Please input the directory to run.')
    print('Available directories:', os.listdir(os.path.join(root, 'generated')))
    exit()

path = os.path.join(root, 'generated', exp)
perf_path = os.path.join(root, 'perf_out', exp)
gs_path = 'gs://tpubenchmarking/' + exp.replace('_b16', '') + '_trace_1.12'
tmp_path = 'gs://tpubenchmarking/tmp'

if not os.path.isdir(perf_path):
  print('Creating new directory: ' + perf_path)
  os.makedirs(perf_path)

files = os.listdir(path)

# sort the files, start with smaller workoads
sfiles = sorted(files, key=lambda x:(int(x.split('-')[0].split('_')[-1]) +\
                                     int(x.split('-')[1].split('_')[-1]) +\
                                     int(x.split('-')[-1].split('_')[-1].split('.')[0])))
c = 0
for f in sfiles:
  if not '.py' in f:
    continue
  err_file = os.path.join(perf_path, f.replace('.py','.err'))
  cmd = 'python ' + os.path.join(path, f) + \
        ' --train_steps=300 --iterations=100 --use_tpu=True --tpu_name=' + os.uname()[1] + ' --model_dir=' + tmp_path
  if os.path.isfile(err_file):
    os.system('grep \"global_step/sec\" ' + err_file + ' > tmp')
    if not os.stat('tmp').st_size == 0:
      continue
    os.system('grep OOM ' + err_file + ' > tmp')
    if not os.stat('tmp').st_size == 0:
      continue
  c += 1
  if c == 10:
    c = 0
    time.sleep(120)
  print('outputs of current wl', err_file)

  os.system('gsutil rm -r ' + tmp_path)
  outfile = open(os.path.join(perf_path, f.replace('.py','.out')), 'w')
  errfile_name = os.path.join(perf_path, f.replace('.py','.err'))
  errfile = open(errfile_name, 'w')
  p = subprocess.Popen(cmd.split(' '), stdout=outfile, stderr=errfile)
  cpt_trace = subprocess.Popen(['python', '/home/wangyu/DL_SynBench/scripts/capture_tpu.py', 
                                os.path.join(gs_path, f.replace('.py', ''))])
  #wd = subprocess.Popen(['python','watch.py', str(p.pid), errfile_name],\
  #                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  #while True: 
  #  time.sleep(10)
  #  print(cpt_trace.poll(),  p.poll())
  #  if cpt_trace.poll() != None or p.poll() != None:
  #    p.kill()
  #    cpt_trace.kill()
  #    break
  p.wait()
  cpt_trace.kill()
  #wd.kill()
print('batch finished!')
