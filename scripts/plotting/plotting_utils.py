import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import seaborn as sns;
import scipy
import os
colors = sns.color_palette("hls", n_colors=11)


'''
Get the range of hyperparameters from a list of labels.
'''
def get_range(labels, p=True):
    keywords = []
    for i in labels[0].split('-'):
        keywords.append(i.split('_')[0])

    values = []
    for i in range(len(keywords)):
        values.append(set())

    for f in labels:
        f_split = f.split('.')[0].split('-')
        for i in range(len(f_split)):
            values[i].add(int(f_split[i].split('_')[1]))
    results = {}
    for i in range(len(keywords)):
      if p == True:
        print(keywords[i], sorted(list(values[i])))
      results[keywords[i]] = sorted(list(values[i]))
    if p == True:
      print('-----------------------')
    return results


'''
Return a list of hyperparameters that are swept from a label.
'''
def get_keyword(l):
    keywords = []
    for i in l.split('-'):
        keywords.append(i.split('_')[0])
    return keywords



'''
Makes d[2] heat maps of dim[0] by dim[1].
'''
def make_heatmap(labels, flops, dim=['node', 'layer', 'bs'], \
                           order=['forward','forward','forward'], r_new={}, title='', lim=[]):
    r = get_range(labels, p=False)
    
    matrix = []
    new = []

    if r_new != {}:
      for k,v in r_new.iteritems():
        r[k] = v
    
    for i in range(len(order)):
        if order[i] == 'backward':
            r[dim[i]] = [j for j in reversed(r[dim[i]])]
        
    keywords = []
    for i in labels[0].split('-'):
        keywords.append(i.split('_')[0])
    
    for k in range(len(r[dim[1]])):
        matrix.append([])
        for i in range(len(r[dim[2]])):
            for j in range(len(r[dim[0]])):
                matrix[-1].append(np.nan)

    for i in range(len(labels)):
        label = labels[i]
        
        l = int(label.split('-')[keywords.index(dim[1])].split('_')[-1])
        n = int(label.split('-')[keywords.index(dim[0])].split('_')[-1])
        b = int(label.split('-')[keywords.index(dim[2])].split('_')[-1])
        
        l_ind = r[dim[1]].index(l)
        n_ind = r[dim[0]].index(n)
        b_ind = r[dim[2]].index(b)
        
        matrix[l_ind][len(r[dim[0]]) * b_ind + n_ind] = flops[i]

    if len(r[dim[2]]) == 1:
        fig, ax = plt.subplots(figsize=(5,5))
    else:
        fig, ax = plt.subplots(figsize=(9,9))
    
    if lim == []:
      im = ax.imshow(matrix)
    else:
      im = ax.imshow(matrix, vmin=lim[0], vmax=lim[1])
    for i in range(1,len(r[dim[2]])):
        ax.plot([i*len(r[dim[0]]) - 0.5, i*len(r[dim[0]]) - 0.5], [-0.5, len(r[dim[1]]) - 0.5], 'k-', color='white')

    ax.set_yticks(range(len(r[dim[1]])))
    ax.set_yticklabels([int(math.log(i, 2)) for i in r[dim[1]]], fontsize=25)
    if dim[1] == 'bs' or dim[1] == 'batchsize':
      ax.set_ylabel(r'Log2(Batch Size)', fontsize=29)
    else:
      ax.set_ylabel(r'Log2(# ' + dim[1].capitalize() + 's)', fontsize=30)

    if dim[0] == 'node':
      node_label = [int(math.log(j, 2)) for j in r[dim[0]]]
      for i in range(len(r[dim[2]])-1):
        for j in r[dim[0]]:
            node_label.append(int(math.log(j, 2)))
    else:
      node_label = r[dim[0]]
    ax.set_xticks(range(len(node_label)))
    ax.set_xticklabels(node_label, fontsize=25)
    if dim[0] == 'bs' or dim[0] == 'batchsize':
      ax.set_xlabel(r'Log2(Batch Size)', fontsize=29)
    elif dim[0] == 'node' or dim[0] == 'layer':
      ax.set_xlabel(r'Log2(# ' + dim[0].capitalize() + 's)', fontsize=30)
    else:
      if dim[0] == 'filtersz':
        ax.set_xlabel("Filters", fontsize=30)
      else:
        ax.set_xlabel(dim[0].capitalize() + 's', fontsize=30)
    
    if len(r[dim[2]]) > 1:
        if dim[2] == 'bs':
          ax.set_title(title + ' (Batchsize: ' + str(r[dim[2]]) + ')', fontsize=15)
        else:
          ax.set_title(title + ' (' + dim[2].capitalize() + ': ' + str(r[dim[2]]) + ')', fontsize=15)
    else:
        ax.set_title(title, fontsize=30)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax= divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=25) 
    return fig


'''
Read a json file, return a dict.
'''
def get_data(filename = ''):
    if not os.path.isfile('../data/' + filename + '.json'):
      print('../data/' + filename + '.json', 'does not exist.')
      return None
    with open('../data/' + filename + '.json', 'r') as infile:
        d = json.load(infile)
    if 'tpu' in filename and 'flops' in d:
        d['flops'] = np.multiply(d['flops'] ,8)
    return d


'''
Filter a dict of data based on range of hyperparameters specified by 'rule'.
'''
def filter_data(rule, data):
    keys = []
    new_data = {}
    for k,v in iter(data.items()):
        if k == 'err_jobs' or k == 'states':
            continue
        new_data[k] = []
        keys.append(k)
    for i in range(len(data['labels'])):
        f = 0
        l = data['labels'][i]
        for j in l.split('-'):
            if j.split('_')[0] in rule and \
                not int(j.split('_')[1]) in rule[ j.split('_')[0]]:
                f = 1
                break
        if f == 0:
            for k in keys:
                new_data[k].append(data[k][i])
    return new_data


'''
Get the speedup of workloads that are in common from a_labels and b_labels.
'''
def get_speedup(a_labels, a_perf, b_labels, b_perf):

    speedups = []
    labels = []
    
    for i in range(len(a_labels)):
        label = a_labels[i]
        if not label in b_labels:
            continue
        ind = b_labels.index(label)
        speedups.append(a_perf[i] / b_perf[ind])
        labels.append(label)
    return labels, speedups


def get_value(l, k=''):
  for i in l.split('-'):
    if k in i:
      return i.split('_')[1] 
  return None

def get_perf(d, k=''):
  ind = d['labels'].index(k)
  return d['example_per_sec'][ind]


def plot_speedup(a_labels, a_perf, b_labels, b_perf, color_dim='layer', x_dim = ['node'], x_flag = ['forward'], title = ''):
    if len(x_dim) > 2:
        print('Not supported: x_dim ', len(x_dim))
        return
    
    r = get_range(a_labels, p=False)
    keywords = []
    for i in a_labels[0].split('-'):
        keywords.append(i.split('_')[0])
    
    color_ind = keywords.index(color_dim)
    x_inds = [keywords.index(i) for i in x_dim] 

    colors = sns.color_palette("hls", n_colors=len(r[color_dim]))
    
    for i in range(len(x_dim)):
        if x_flag[i] == 'reverse':
            r[x_dim[i]] = [j for j in reversed(r[x_dim[i]])]

    speedups = []
    labels = []
    x = []
    
    for i in range(len(a_labels)):
        label = a_labels[i]
        if not label in b_labels:
            continue
      
        dims = [int(label.split('-')[j].split('_')[-1]) for j in x_inds]
        ind1 = r[x_dim[0]].index(dims[0])
        if len(x_dim) == 2:
            ind2 = r[x_dim[1]].index(dims[1])
        else:
            ind2 = 0
        ind = b_labels.index(label)
        
        speedups.append(a_perf[i] / b_perf[ind])
            
        labels.append(label)
        x.append(ind2 * len(r[x_dim[0]]) + ind1)
        
    print('length of results', len(x))

    fig, ax = plt.subplots(figsize=(7,3))

    legend_flag = {}
    
    for i in range(len(x)):
        l = int(labels[i].split('-')[color_ind].split('_')[-1])
        if not l in legend_flag:
            ax.plot(x[i], speedups[i], '.', color = colors[r[color_dim].index(l)], label = color_dim + '-' + str(l))
            legend_flag[l] = 1
        else:
             ax.plot(x[i], speedups[i], '.', color = colors[r[color_dim].index(l)])

    
    ax.legend(frameon=True, fontsize=15)
    
    if len(x_dim) == 1:
        l = [str(i) for i in r[x_dim[0]]]
    else:
        l = []
        for i in r[x_dim[1]]:
            for j in r[x_dim[0]]:
                l.append(str(i) + '-' + str(j))
        
    l.append('Avg')
    ax.set_xticks(range(len(l)))

    if len(x_dim) == 1:
        ax.set_xticklabels(l)
        ax.set_xlabel(x_dim[0], fontsize = 20)
    else:
        ax.set_xticklabels(l, rotation=65)
        ax.set_xlabel(x_dim[1] + '-' + x_dim[0], fontsize = 20)
        
    ax.plot([len(l)-1], np.mean(speedups), '*', color = 'r')
    print('avg speedup:', np.mean(speedups))
    print(np.sum(np.less(speedups, 1)), np.sum(np.less(speedups, 1))*1.0/len(speedups), 'models have speedups < 1.')
    
    ax.set_ylabel('Speedup', fontsize = 20)
    ax.set_title(title, fontsize = 20)
    ax.grid()
 
    handles, labels = ax.get_legend_handles_labels()
    labels = [int(i.split('-')[-1]) for i in labels]
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    labels = [color_dim + '-' + str(i) for i in labels]
    ax.legend(handles, labels, frameon=True, loc=2, bbox_to_anchor=(1, 1),
              fancybox=False, shadow=False, ncol=1)


def speedup_params(a_labels, a_perf, b_labels, b_perf, b_params, legend_box=(), loc='', marker='', lim=[], color='', ncol=1, title = ''):
    markers = ['.', '^', '*', '+', 'x', 'd', '>']
    speedups = []
    p = []
    l = []
    for i in range(len(a_labels)):
        label = a_labels[i]
        if not label in b_labels:
            continue
        ind = b_labels.index(label)
        speedups.append(a_perf[i] / b_perf[ind])
        p.append(b_params[ind])
        l.append(label)
    print('length of speedups', len(speedups))
    print(max(speedups), min(speedups))
    fig, ax = plt.subplots(figsize=(3,3))
    if color == '':
      ax.plot(p, speedups, '.')
    else:
      f = {}
      k = get_keyword(l[0])
      k_ind = k.index(color)
      r = get_range(l, p=False)
      num_color = len(r[color])
      colors = sns.color_palette("hls", n_colors=num_color)
      for i in range(len(l)):
        n = int(l[i].split('-')[k_ind].split('_')[-1])
        if n in f:
          if marker == '':
            ax.plot(p[i], speedups[i], marker=markers[f[n]], color=colors[f[n]])
          else:
            ax.plot(p[i], speedups[i], marker=marker, color=colors[f[n]])
        else:
          f[n] = r[color].index(n)
          if marker == '':
            ax.plot(p[i], speedups[i], marker=markers[f[n]], color=colors[f[n]], label= str(n))
          else:
            ax.plot(p[i], speedups[i], marker=marker, color=colors[f[n]], label= str(n))
      handles, labels = ax.get_legend_handles_labels()
      labels = [int(i) for i in labels]
      labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
      if color == 'batchsize':
        color='bs'
      if color == 'embeddingsize':
        color='embed'
      if color == 'block':
        color = 'blk'
      if color == 'filtersz':
        color = 'filter'
      labels = [color + '-' + str(i) for i in labels]
      new_labels = []
      for l in labels:
        if '1024' in l:
          new_labels.append(l.replace('1024', '1k'))
        elif '2048' in l:
          new_labels.append(l.replace('2048', '2k'))
        elif '4096' in l:
          new_labels.append(l.replace('4096', '4k'))
        elif '8192' in l:
          new_labels.append(l.replace('8192', '8k'))
        elif '16384' in l:
          new_labels.append(l.replace('16384', '16k'))
        else:
          new_labels.append(l)

      if loc != '':
        ax.legend(handles, new_labels, loc=loc, frameon=True, ncol=ncol, fontsize=12)
      else:
        ax.legend(handles, new_labels, frameon=True, ncol=ncol, fontsize=12)
      if legend_box != ():
         ax.legend(handles, new_labels, frameon=True, ncol=ncol, fontsize=12, bbox_to_anchor=legend_box)
    plt.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    left, right = ax.get_xlim()
    ax.set_xlim([left, right])
    ax.plot([left, right], [1, 1], 'k-', color='orange')
    if lim != []:
      ax.set_ylim(lim)
    ax.set_xlabel('Params', fontsize = 25)
    ax.set_ylabel('Speedups', fontsize = 25)
    ax.set_title(title, fontsize = 25)
    return fig
      

def get_n_from_label(l, dim):
    if dim == 'op':
        return l.split('_')[-1].strip('0123456789-')
    for i in l.split('-'):
        if dim in i:
            return i.split('_')[-1]
    return None


def remove_while_op(d):
    new_d = {}
    keys = d.keys()
    for k in keys:
        new_d[k] = []
    for i in range(len(d['labels'])):
        if not 'while' in d['labels'][i]:
            for k in keys:
                new_d[k].append(d[k][i])
    return new_d


def plot_roofline(f, ax, d, tpu_peak, membdw_peak, \
                  scale='absolute', color_map={}, color_dim='', color=0, thre=1, label='', title=''):

    colormap = {}
    if scale == 'absolute':
      flops = np.multiply(d['flops_perc'], tpu_peak/100)
    else:
      flops = d['flops_perc']

    labels = d['labels']
    intensity = d['intensity']
    time = d['time_perc']
    if color_dim == '':
        if color == 0:
          ax.plot(d['intensity'], flops, '.', label=label)
        else:
          ax.plot(d['intensity'], flops, '.', label=label, color=color, alpha=0.9)
    else:
        
        hist = {}
        for i in range(len(labels)):
            l = d['labels'][i]
            n = get_n_from_label(l, color_dim)
            if time[i] < thre:
              continue
            if intensity[i]<=0 or flops[i]<=0:
              continue
            if not n in hist:
                hist[n] = 0    
            hist[n] += 1
        for k,v in iter(hist.items()):
            hist[k] = v*1.0/len(labels)
        
        m = {}
        mycolors = sns.color_palette("hls", n_colors=len(hist)+2)
        for i in range(len(labels)):
            if time[i] < thre:
              continue
            if intensity[i]<=0 or flops[i]<=0:
              continue
            l = labels[i]
            n = get_n_from_label(l, color_dim)
                    
            if color_map != {}:
                if n in color_map:
                    if n in m:
                      ax.plot(intensity[i], flops[i], '.', color=color_map[n], marker='.')
                    else:
                      ax.plot(intensity[i], flops[i], '.', color=color_map[n], marker='.', label = n)
                      m[n] = 1
                continue    
            if n in m:
                ax.plot(intensity[i], flops[i], '.',
                        color=mycolors[m[n]], marker='.')
            elif not n in m:
                
                m[n] = len(m) % len(colors)
                colormap[n] = mycolors[m[n]]
                ax.plot(intensity[i], flops[i], '.',
                        color=mycolors[m[n]], label = n, 
                        #markeredgecolor='black', markeredgewidth=0.5, 
                        marker='.')
        if color_dim != 'op':
            handles, ls = ax.get_legend_handles_labels()
            ls = [int(i) for i in ls]
            ls, handles = zip(*sorted(zip(ls, handles), key=lambda t: t[0]))
            ls = [color_dim + '-' + str(i) for i in ls]
            ax.legend(handles, ls, frameon=True)
        else:
            ax.legend(frameon=True, bbox_to_anchor=(1, 0.5))

    x1 = tpu_peak / membdw_peak
    y1 = tpu_peak
      
    if max(d['intensity']) > x1:
        if color == 0:
            ax.hlines(y=y1, xmin=x1, 
                xmax=max(d['intensity']), linewidth=2, color=colors[0])
        else:
            ax.hlines(y=y1, xmin=x1, 
                xmax=max(d['intensity']), linewidth=2, color=color)
    
    x2 = min(d['flops_perc'])*(tpu_peak/100)/membdw_peak
    y2 = min(d['flops_perc'])*(tpu_peak/100)

    if scale == 'relative':
        y1 = 100
        y2 = x2 * membdw_peak / tpu_peak * 100
    if color == 0:
        ax.plot([x1, x2], [y1, y2], linewidth=2, color=colors[0])
    else:
        ax.plot([x1, x2], [y1, y2], linewidth=2, color=color)
        
    ax.set_yscale('log')
    ax.set_xscale('log')
    if scale == 'absolute':
      ax.set_ylabel('GFLOPS', fontsize=15)
    else:
      ax.set_ylabel('FLOPS %', fontsize=15)
      ax.set_ylim(top=100)
    ax.set_xlabel('Floating Ops/Byte', fontsize=15)
    ax.set_title(title, fontsize=15)

    if colormap == {}:
        colormap = color_map
    return f, ax, colormap


def plot_twodim(label1, perf1, label8, perf8, same_lim=True, color_dim='', xlabel='', ylabel='', label='', scale='', lim=[], title=''):

    l_colors = sns.color_palette("hls", n_colors=10)

    f, ax = plt.subplots(figsize=(3,3))

    m1 = 0
    m2 = 0

    m = {}
    color_map = {}
    for i in range(len(label1)):
            l = label1[i]

            if not l in label8:
                continue
            ind = label8.index(l)

            n = get_value(l, color_dim)

            if m1 < perf1[i]:
                m1 = perf1[i]
            if m2 < perf8[ind]:
                m2 = perf8[ind]

            if color_dim == '':
                ax.plot(perf1[i], perf8[ind], '.', color=colors[0], label=label)

            else:

              if not n in m:
                m[n] = len(m)
                color_map[n] = l_colors[m[n]]
                ax.plot(perf1[i], perf8[ind], '.', color=l_colors[m[n]], label=n)
              else:
                ax.plot(perf1[i], perf8[ind], '.', color=l_colors[m[n]])

    if color_dim != '':
        handles, ls = ax.get_legend_handles_labels()
        ls = [int(i) for i in ls]
        ls, handles = zip(*sorted(zip(ls, handles), key=lambda t: t[0]))
        if color_dim == 'batchsize':
            color_dim = 'bs'
        ls = [color_dim + '-' + str(i) for i in ls]
        ax.legend(handles, ls, frameon=True, fontsize=12, bbox_to_anchor=(1.05, 1.05))

    if same_lim:
        max_perf = max([max(perf1), max(perf8)])
        min_perf = min([min(perf1), min(perf8)])
        if lim != []:
          max_perf = lim[1]
          min_perf = lim[0]
        ax.plot([min_perf,max_perf], [min_perf,max_perf], '--')
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    if scale == 'log':
        ax.set_yscale('log')
        ax.set_xscale('log')
    plt.tick_params(axis='y', which='major', labelsize=11)
    plt.tick_params(axis='x', which='major', labelsize=11)

    if lim != []:
      ax.set_xlim(lim)
      ax.set_ylim(lim)
    ax.set_title(title, fontsize=15)
    return f, color_map
