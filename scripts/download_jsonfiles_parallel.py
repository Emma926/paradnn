import os
import sys
import multiprocessing
import tempfile

trace_name= sys.argv[1]
bucket = sys.argv[2]

download_path = '../output/' + trace_name + '/'
if not os.path.isdir(download_path):
  os.mkdir(download_path)
else:
  os.system('mv ' + download_path + ' ../output/' + trace_name + '_old')
  os.mkdir(download_path)

gs_path = 'gs://' + bucket + '/' + trace_name + '/'

def do_work(wl):
  wl = wl.strip('\n')
  if wl == gs_path:
    return

  cur_path = os.path.join(download_path, wl.split('/')[-2])
  if not os.path.isdir(cur_path):
    os.mkdir(cur_path)
  
  with tempfile.NamedTemporaryFile() as ftmp:
    os.system('gsutil ls ' + os.path.join(wl, 'plugins', 'profile') + ' > ' + ftmp.name)
    
    for sd in ftmp:
      sd = sd.strip('\n')
      if sd == os.path.join(wl, 'plugins', 'profile') + '/':
        continue
      cur_path = os.path.join(download_path, wl.split('/')[-2], sd.split('/')[-2])
      if not os.path.isdir(cur_path):
        os.mkdir(cur_path)  
    
      cp_cmd = 'gsutil cp ' + sd + '*input_pipeline.json ' +  cur_path
      os.system(cp_cmd)
      cp_cmd = 'gsutil cp ' + sd + '*op_profile.json ' +  cur_path
      os.system(cp_cmd)
      cp_cmd = 'gsutil cp ' + sd + '*overview_page.json ' +  cur_path
      os.system(cp_cmd)

with tempfile.NamedTemporaryFile() as ftmp:
  os.system('gsutil ls ' + gs_path + ' > ' + ftmp.name)
  p = multiprocessing.Pool(64)
  try:
    res = p.map_async(do_work, list(ftmp))
    while not res.ready():
      try:
        res.get(1)
      except multiprocessing.TimeoutError:
        pass
  except KeyboardInterrupt:
    p.terminate()
  else:
    p.close()
  p.join()
