from sklearn import linear_model
from sklearn import preprocessing
from sklearn.utils import shuffle
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns;
colors = sns.color_palette("hls", n_colors=11)


def create_training_data(labels, speedups, logistic=False, order = 2):
    x = []
    y = []
    if not order in [1,2]:
        print("order = ", order, "is not supported.")
        
    for i in range(len(labels)):
        l = labels[i]
        x.append([])
        for j in l.split('-'):
            x[-1].append(int(j.split('_')[-1]))

        tmp = x[-1][:]
        if order == 2:
            for a in range(len(tmp)):
                for b in range(a, len(tmp)):
                    x[-1].append(tmp[a]*tmp[b])
            
        if logistic == True:
            if speedups[i] > 1:
                y.append(1)
            else:
                y.append(0)
        else:
            y.append(speedups[i])

    features = []
    for i in labels[0].split('-'):
        features.append(i.split('_')[0])
    tmp = features[:]
    if order == 2:
        for a in range(len(tmp)):
            for b in range(a, len(tmp)):
                features.append(tmp[a] + ' * ' + tmp[b])

    x = preprocessing.scale(x)
    if logistic == False:
        y = preprocessing.scale(y)
    return x, y, features


def get_top_features(coef, features, ylim=[], thre=0.1, title=''):
    f_feat = []
    f_coef = []

    for i in range(len(coef)):
        if abs(coef[i]) > thre:
            f_feat.append(features[i])
            f_coef.append(coef[i])
            
    s_feat = [x for _,x in sorted(zip(f_coef,f_feat))]
    s_coef = sorted(f_coef)
    
    fig, ax = plt.subplots(figsize=(3,2))
    ax.barh(range(len(s_coef)), s_coef)
    ax.set_yticks(range(len(s_coef)))
    if 'bs' in s_feat:
      s_feat[s_feat.index('bs')] = 'batchsize'
    ax.set_yticklabels([i.capitalize().replace('size','').replace('sz','') for i in s_feat], fontsize=20)
    ax.set_xlabel('LR Weights', fontsize=20)
    ax.grid()
    if ylim == []:
      ax.set_xlim([-1, 1])
    else:
      ax.set_xlim(ylim)
    ax.set_title(title, fontsize=20)
    plt.tick_params(axis='x', which='major', labelsize=13)
    return s_feat, s_coef, fig

def predict_with_features(features, features_sel, coef, x, y, plot=False):
    ind = []
    for i in range(len(features)):
        if features[i] in features_sel:
            ind.append(i)
    
    a = []
    err = []
    for i in range(len(x)):
        t = 0
        for j in range(len(ind)):
            t += x[i][ind[j]]*coef[ind[j]]
        a.append(t)
        err.append(abs(t - y[i]))
    if plot == True:
        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(a, y[:], '.')
        ax.plot([-1.5, 1.5], [-2, 2], 'k-', color='black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Real')
    return np.mean(err)
    

def regression(labels, speedups, order=1, ylim=[], plot=False, title=''):
    x, y, features = create_training_data(labels, speedups, order = order)
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    coef = regr.coef_
    features_sel, coef_sel, fig = get_top_features(coef, features, ylim=ylim, thre = 0.0, title=title)
    mae = predict_with_features(features, features_sel, coef, x, y, plot)
    return fig


def regression_all(labels_t, perf_t, labels_g, perf_g, labels_c=[], perf_c=[], xlim = [], title=''):
    x, y, features_t = create_training_data(labels_t, perf_t, order = 1)
    regr_t = linear_model.LinearRegression()
    regr_t.fit(x, y)
    coef_t = regr_t.coef_

    x, y, features_g = create_training_data(labels_g, perf_g, order = 1)
    regr_g = linear_model.LinearRegression()
    regr_g.fit(x, y)
    coef_g = regr_g.coef_

    nocpu = False
    if len(labels_c) == 0 :
      features_c = features_t[:]
      coef_c = [0] * len(features_t)
      nocpu = True
    else: 
      x, y, features_c = create_training_data(labels_c, perf_c, order = 1)
      regr_c = linear_model.LinearRegression()
      regr_c.fit(x, y)
      coef_c = regr_c.coef_

    for i in range(len(features_t)):
      if features_t[i] == 'bs':
        features_t[i] = 'batchsize'
    
    if nocpu == True:
      fig, ax = plt.subplots(1,2,figsize=(5, 2))
    else:
      fig, ax = plt.subplots(1,3,figsize=(7, 2))
    cnt = 0

    if nocpu == True:
      for i in range(2):
        ax[i].grid()
        ax[i].set_xlim(xlim)
    else:
      for i in range(3):
        ax[i].grid()
        ax[i].set_xlim(xlim)

    if nocpu == False:
      ax[cnt].barh(range(len(coef_c)), [i for i in reversed(coef_c)])
      ax[cnt].set_xlabel('LR Weights (CPU)', fontsize=14)
      ax[cnt].set_yticks(range(len(coef_c)))
      ax[cnt].set_yticklabels([i.capitalize() for i in reversed(features_t)], fontsize=14)
      cnt += 1

    ax[cnt].barh(range(len(coef_g)), [i for i in reversed(coef_g)])
    ax[cnt].set_xlabel('LR Weights (GPU)', fontsize=14)
    ax[cnt].set_yticks(range(len(coef_g)))
    if nocpu == False:
      ax[cnt].set_yticklabels(['' for i in features_t])
    else:
      ax[cnt].set_yticklabels([i.capitalize() for i in reversed(features_t)], fontsize=14)
    
    cnt += 1

    ax[cnt].barh(range(len(coef_t)), [i for i in reversed(coef_t)])
    ax[cnt].set_xlabel('LR Weights (TPU)', fontsize=14)
    ax[cnt].set_yticks(range(len(coef_t)))
    ax[cnt].set_yticklabels(['' for i in features_t])
  
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    return fig

    
def classification(labels, speedups, order=1):
    x, y, features = create_training_data(labels, speedups, logistic=True, order = order)
    print(sum(y)*1.0/len(y))
    regr = linear_model.LogisticRegression()
    regr.fit(x, y)
    coef = regr.coef_[0]
    features_sel, coef_sel = get_top_features(coef, features, thre = 0.0)
    pred = regr.predict(x)
    r = regr.score(x, y)
    return r
