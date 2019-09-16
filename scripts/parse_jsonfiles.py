import os
import json
import copy
import numpy as np
import datetime

trace_name= sys.argv[1]

input_path = '../output/' + trace_name

def read_json(f):
    with open(f, 'r') as infile:
        d = json.load(infile)
    return d

def traverse_op_tree(f):
    import Queue
    op_flops = {'matrix': set(), 'vector': set()}
    op_names = {'matrix': set(), 'vector': set()}
    op_category = {'matrix': set(), 'vector': set()}

    d = read_json(f)
    q = Queue.Queue()
    q.put(d['byCategory'])
    while not q.empty():
      c = q.get()
      if c['children'] <> []:
        for i in c['children']:
          q.put(i)
      else:
        # fused ops do not have individual metrics
        # problem: can't breakdown when vector and matrix ops are fused together
        if not 'metrics' in c:
          continue
        f = c['metrics']['flops'] / c['metrics']['time']
        n = c['name'].split('.')[0]
        cat = c['xla']['category'].lower()
        pro = c['xla']['provenance'].split('/')[-1].lower()
        if cat == 'convolution':
          t = 'matrix'
        else:
          t = 'vector'
        op_flops[t].add(f)
        op_names[t].add(n)
        op_category[t].add(cat)
    print(op_names)
    print(op_category)
    print(op_flops)
    return op_flops

def parse_op_profile(f, pr = True):
    d = read_json(f)
    # get op time and flops breakdown
    # overall TPU flops utilization 
    op = {}
    # time%, flops%, memorybandwidth%, op intensity 
    flops_perc = d['byCategory']['metrics']['flops']
    mem_perc = d['byCategory']['metrics']['memoryBandwidth']
    rawMB = d['byCategory']['metrics']['rawBytesAccessed']/1e6
    op_intensity = d['byCategory']['metrics']['rawFlops']*1.0/d['byCategory']['metrics']['rawBytesAccessed'] 
    op['overall'] = (1, flops_perc, mem_perc, rawMB, op_intensity)
    if pr == True:
        print('Op Profile'
        print('op name: time%, flops%, memorybandwidth%, rawMB, op intensity'
        print('overall:', op['overall']

    # op name, time %, flops
    for i in d['byCategory']['children']:

      time_perc = i['metrics']['time']
      flops_perc = i['metrics']['flops'] / time_perc
      mem_perc = i['metrics']['memoryBandwidth']
      rawMB = i['metrics']['rawBytesAccessed'] / 1e6
      
      if i['metrics']['rawBytesAccessed'] == 0:
        op_intensity = float('nan')
      else:
        op_intensity = i['metrics']['rawFlops']*1.0/i['metrics']['rawBytesAccessed'] 
      op[i['name']] = (time_perc, flops_perc, mem_perc, rawMB, op_intensity)

      if pr == True:
        print(i['name'], op[i['name']], i)
    return op

def parse_input_pipeline(f, version=1.8, pr=True):
    d = read_json(f)
    m = float(d[0]['p']['infeed_percent_average'])
    std = float(d[0]['p']['infeed_percent_standard_deviation'])
    if pr == True:
        print('Infeed Time and Std')
        print(m, std)
        print('Host Time Breakdown')
        
    kUsPerMs = 1000.0
    hosttime = {}
    t = 0
    
    if version == 1.8:
        for k,v in d[2]['p'].iteritems():
            hosttime[k[:-3]] = float(v)/kUsPerMs
            t += hosttime[k[:-3]]
    elif version == 1.7:
        for k,v in d[1]['p'].iteritems():
            hosttime[k[:-3]] = float(v)/kUsPerMs
            t += hosttime[k[:-3]]
        
    if t <> 0:
        for k,v in hosttime.iteritems():
            hosttime[k] = hosttime[k] / t
            if pr == True:
                print(k, hosttime[k])
    
    return (m, std), hosttime

def get_top_op(op, top=3):
    import copy
    op_cp = copy.copy(op)
    results = []
    for i in range(top):
        best_op = 0
        best_time = 0
        for k,v in op.iteritems():
            if k == 'overall':
                continue
            if v[0] > best_time:
                best_op = k
                best_time = v[0]
        if best_time > 0:
            results.append((best_op, op[best_op]))
            op[best_op] = (0,0)
        else:
            return results
    return results

wls = os.listdir(input_path)
label_all = []
time_all = []
flops_all = []
flops_m_all = []
flops_std_all = []
flops_trace_all = []
membdw_all = []
rawMB_all = []
op_intensity_all = []
infeed_all = []
infeed_m_all = []
infeed_std_all = []
infeed_trace_all = []

label_breakdown = []
time_breakdown = []
flops_breakdown = []
flops_m_breakdown = []
flops_std_breakdown = []
membdw_breakdown = []
rawMB_breakdown = []
op_intensity_breakdown = []
infeed_breakdown = []
infeed_m_breakdown = []
infeed_std_breakdown = []

for wl in wls:
  traces = os.listdir(os.path.join(input_path, wl))
  s_traces = sorted(traces, key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d_%H:%M:%S'))
  flops = 0
  flops_l = []
  membdw = 0
  op_intensity = 0
  infd = 0
  infd_std = 0
  infd_l = []

  best_op = {}

  for trace in s_traces:
    cwd = os.path.join(input_path, wl, trace)
    op = None
    for f in os.listdir(cwd):
      if 'op_profile' in f:
        op = parse_op_profile(os.path.join(cwd, f), False)
        #op_flops = traverse_op_tree(os.path.join(cwd, f))
      elif 'input_pipeline' in f:
        infeed, host = parse_input_pipeline(os.path.join(cwd, f), version=1.8, pr=False)
    if op <> None:
      flops_l.append(op['overall'][1] * 100)
      infd_l.append(infeed[0])
    if op <> None and op['overall'][1] * 100 > flops:
        time =  op['overall'][0] * 100
        flops = op['overall'][1] * 100
        membdw = op['overall'][2] * 100
        rawMB = op['overall'][3]
        op_intensity = op['overall'][4]
        infd = infeed[0]
        infd_std = infeed[1]
        best_op = copy.deepcopy(op)

  if flops == 0:
    continue
  
  label_all.append(wl)
  time_all.append(time)
  flops_all.append(flops)
  flops_trace_all.append(flops_l)
  flops_m_all.append(np.mean(flops_l))
  flops_std_all.append(np.std(flops_l))
  membdw_all.append(membdw)
  rawMB_all.append(rawMB)
  op_intensity_all.append(op_intensity)
  infeed_all.append(infd)
  infeed_m_all.append(np.mean(infd_l))
  infeed_std_all.append(infd_std)
  infeed_trace_all.append(infd_l)
  
  for k,v in best_op.iteritems():
    if k == 'overall':
      continue
    label_breakdown.append(wl + '-' + k)
    time_breakdown.append(v[0] * 100)
    flops_breakdown.append(v[1] * 100)
    membdw_breakdown.append(v[2] * 100)
    rawMB_breakdown.append(v[3])
    op_intensity_breakdown.append(v[4])


print(label_all)
print(flops_all)

d = {
  'labels': label_all,
  'time_perc': time_all,
  'flops_perc': flops_all,
  'flops_trace': flops_trace_all,
  'flops_perc_m': flops_m_all,
  'flops_perc_std': flops_std_all,
  'memory_bandwidth': membdw_all,
  'rawMB': rawMB_all,
  'intensity': op_intensity_all,
  'infeed': infeed_all,
  'infeed_m': infeed_m_all,
  'infeed_std': infeed_std_all,
  'infeed_trace': infeed_trace_all,
}

print(len(label_all))
name = trace_name
if os.path.isfile('data/' + name + '.json'):
  with open('data/' + name + '.json','r') as infile:
    din = json.load(infile)
    print(len(label_all) - len(din['labels']), 'more data points than last time.')

with open('data/' + name + '.json','w') as outfile:
    json.dump(d, outfile)
    print('Results written in data/' + name + '.json')

d = {
  'labels': label_breakdown,
  'time_perc': time_breakdown,
  'flops_perc': flops_breakdown,
  'memory_bandwidth': membdw_breakdown,
  'intensity': op_intensity_breakdown,
  'rawMB': rawMB_breakdown,
}

name = trace_name + '_opbreakdown'
if os.path.isfile('data/' + name + '.json'):
  with open('data/' + name + '.json','r') as infile:
    din = json.load(infile)
    print(len(label_breakdown) - len(din['labels']), 'more data points than last time.')

with open('data/' + name + '.json','w') as outfile:
    json.dump(d, outfile)
    print('Results written in data/' + name + '.json')
