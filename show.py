import json
import math

from pprint import pprint
from collections import defaultdict

from sklearn import cluster, mixture
import numpy as np

from itertools import cycle

import matplotlib.pyplot as plt
import os

def rotate_flat(locs, tee, flag):
  tex, tey = tee
  fgx, fgy = flag

  if fgx < tex:
    xconvert = -1
  else:
    xconvert = 1
  if fgy < tey:
    yconvert = -1
  else:
    yconvert = 1

  res = [[ ((x-tex)*xconvert) / 3.0, ((y-tey)*yconvert) / 3.0] for x,y in locs]
  fgx = ((fgx-tex)*xconvert) / 3.0
  fgy = ((fgy-tey)*yconvert) / 3.0
  tex = 0
  tey = 0

  tee_flag_angle = np.arctan2(fgy - tey, fgx - tex)
  desired_angle = np.radians(0)# math.pi/6 #30 degrees
  rot_angle = desired_angle - tee_flag_angle

  c, s = np.cos(desired_angle), np.sin(desired_angle)

  fgxnew = fgx * c - fgy * s
  fgynew = fgx * s + fgy * c

  res2 = []
  for px, py in res:
    xnew = px * c - py * s
    ynew = px * s + py * c
    res2.append([ynew - fgynew, xnew - fgxnew])

  if tee_flag_angle > desired_angle:
    pass
  else:
    pass

  return res2

def gather_shots(hole_number, round_numbers=[], shot_number=1):
  data_path = "%s/data/%s/" % (os.getcwd(), str(hole_number))
  for round_number in round_numbers:
    filename = "%s_%s.json" % (str(round_number), str(hole_number))
    filepath = data_path + filename
    with open(filepath) as f:
      data = json.load(f)

    fg = data['Rs'][0]['Hs'][0]['Fg']
    fgx = fg['X']
    fgy = fg['Y']
    flag = [fgx, fgy]

    te = data['Rs'][0]['Hs'][0]['Te']
    tex = te['X']
    tey = te['Y']

    tee = [tex, tey]

    hps = data['Rs'][0]['Hs'][0]['HPs']
    sks = []
    for hp in hps:
      sk = hp['Sks'][shot_number - 1]
      skx = sk['X']
      sky = sk['Y']
      sks.append([skx, sky])

  return rotate_flat(sks, tee, flag)

def calc_center(group_locs):
  a = np.array(group_locs)
  center = np.mean(a, axis=0)
  return center


def plot_clusters(ax, ncs, locs, labels, centers, title):
  num_found_clusters = set(labels)
  colors = cycle('bgrcymbgrcym')
  for k, color in zip(num_found_clusters, colors):
    class_members = labels == k

    group_locs = [x for x, y in zip(locs, class_members) if y]

    xs = [val[0] for val in group_locs]
    ys = [val[1] for val in group_locs]

    if k == -1: #outliers!
      color = 'k'

    ax.plot(xs, ys, color + 'o', ms=3)
    if centers is not None and centers[k] is not None:
      ax.plot(centers[k][0], centers[k][1], 'kx', ms=10)

  ax.set_title(title, size=8)
  ax.tick_params(axis='both', which='major', labelsize=8)
  return plt

def create_fig_axes(num_graphs, num_graph_cols=3):

  num_rows = num_graphs / num_graph_cols
  num_empty = 0
  if num_graphs % num_graph_cols != 0:
    num_rows += 1
    num_empty = num_graph_cols - (num_graphs % num_graph_cols)

  if num_graphs < num_graph_cols:
    num_graph_cols = num_graphs

  fig, axes = plt.subplots(num_rows, num_graph_cols)

  # Make sure we don't have empty space or empty graphs
  if num_graphs == 1:
    axes = np.array([axes])
  if num_graph_cols == 1:
    axes = np.array([[axes[i]] for i in range(num_graphs)])
  if len(axes.shape) == 1:
    axes = np.array([axes])
  for i in range(num_empty):
    fig.delaxes(axes[-1][i*-1-1])

  return fig, axes

def further_plot(results, title, param_label, num_graph_cols=2, save=False, save_path=None):

  num_graphs = len(results)
  fig, axes = create_fig_axes(num_graphs, num_graph_cols=num_graph_cols)

  for index, (centers, labels, ncs) in enumerate(results):
    ax_ind_0 = index / num_graph_cols
    ax_ind_1 = index % num_graph_cols

    subtitle = '%s=%s' % (param_label, str(ncs))

    plot_clusters(axes[ax_ind_0][ax_ind_1], ncs, locs, labels, centers, subtitle)

  fig.suptitle(title, size=14)
  fig.subplots_adjust(wspace=0.4, hspace=0.4)
  if save:
    fig.savefig(save_path, dpi=1000)
  else:
    plt.show()

  return fig

def k_means_process(locs, num_clusterss, num_graph_cols=2, save=False):

  title = 'K Means, hole %s, shot %s, round %s' % (str(hole_number), str(shot_number), ','.join(map(str, round_numbers)))
  save_path = "imgs/k_means/hole_%s_shot_%s_clusters_%s.png" % (str(hole_number), str(shot_number), '4')

  results = []
  for ncs in num_clusterss:
    centers, labels, inertia = cluster.k_means(locs, n_clusters=ncs)
    results.append([centers, labels, ncs])

  fig = further_plot(results, title, 'n_clusters', num_graph_cols, save=save, save_path=save_path)

  return results

def db_scan_process(locs, epss, num_graph_cols=3, save=False):

  results = []
  for eps in epss:
    dbs = cluster.DBSCAN(eps=eps).fit(locs)
    labels = dbs.labels_
    centers = []
    num_found_clusters = set(labels)
    for val in num_found_clusters:
      if val == -1:
        center = None
      else:
        class_members = labels == val
        group_locs = [x for x, y in zip(locs, class_members) if y]
        center = calc_center(group_locs)
      centers.append(center)
    results.append([centers, labels, eps])

  title = 'DBSCAN, hole %s, shot %s, round %s' % (str(hole_number), str(shot_number), ','.join(map(str, round_numbers)))
  save_path = "imgs/dbscan/shot_%s_eps_%s_clusters_%s.png" % (str(shot_number), str(eps), str(len(num_found_clusters)))

  fig = further_plot(results, title, 'eps', num_graph_cols, save=save, save_path=save_path)

  return results

def mean_shift_process(locs, bandwidths, num_graph_cols=3, save=False):

  title = 'Mean Shift, hole %s, shot %s, round %s' % (str(hole_number), str(shot_number), ','.join(map(str, round_numbers)))
  save_path = "imgs/mean_shift/hole_%s_shot_%s.png" % (str(hole_number), str(shot_number))

  results = []
  for bw in bandwidths:
    ms = cluster.MeanShift(bandwidth=bw).fit(locs)
    labels = ms.labels_
    centers = ms.cluster_centers_
    results.append([centers, labels, bw])

  fig = further_plot(results, title, 'bandwidth', num_graph_cols=num_graph_cols, save=save, save_path=save_path)

  return results

def agglomerative_clustering_process(locs, num_clusters, num_graph_cols=3, save=False):

  title = 'Agglomerative Clustering, hole %s, shot %s, round %s' % (str(hole_number), str(shot_number), ','.join(map(str, round_numbers)))
  save_path = "imgs/agg_clustering/hole_%s_shot_%s_clusters_%s.png" % (str(hole_number), str(shot_number), str(num_clusters))

  results = []
  for ncs in num_clusterss:
    ac = cluster.AgglomerativeClustering(n_clusters=ncs).fit(locs)
    labels = ac.labels_

    centers = []
    num_found_clusters = set(labels)
    for val in num_found_clusters:
      class_members = labels == val
      group_locs = [x for x, y in zip(locs, class_members) if y]
      center = calc_center(group_locs)
      centers.append(center)

    results.append([centers, labels, ncs])

  fig = further_plot(results, title, 'n_clusters', num_graph_cols=num_graph_cols, save=save, save_path=save_path)

  return results


if __name__ == '__main__':

  hole_number = 10
  shot_number = 1
  round_numbers = [1]
  num_clusterss = [2, 3, 4, 5, 6, 7]
  num_graph_cols = 3
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  k_means_process(locs, num_clusterss, num_graph_cols=num_graph_cols)


  hole_number = 10
  shot_number = 2
  round_numbers = [1]
  num_clusterss = [2, 3, 4, 5, 6, 7]
  num_graph_cols = 3
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  k_means_process(locs, num_clusterss, num_graph_cols=num_graph_cols)


  hole_number = 10
  shot_number = 1
  round_numbers = [1]
  epss = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]
  num_graph_cols = 3
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  db_scan_process(locs, epss, num_graph_cols=num_graph_cols)

  hole_number = 10
  shot_number = 2
  round_numbers = [1]
  epss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
  num_graph_cols = 3
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  db_scan_process(locs, epss, num_graph_cols=num_graph_cols)


  hole_number = 10
  shot_number = 1
  round_numbers = [1]
  num_clusterss = [2, 3, 4, 5, 6, 7]
  num_graph_cols = 2
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  agglomerative_clustering_process(locs, num_clusterss)


  hole_number = 10
  shot_number = 2
  round_numbers = [1]
  num_clusterss = [2, 3, 4, 5, 6, 7]
  num_graph_cols = 3
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  agglomerative_clustering_process(locs, num_clusterss)

  hole_number = 10
  shot_number = 1
  round_numbers = [1]
  bandwidths=[5, 10, 15, 20, 25, 30, 35, 40, 45]
  num_graph_cols = 2
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  mean_shift_process(locs, bandwidths)

  hole_number = 10
  shot_number = 2
  round_numbers = [1]
  bandwidths=[10, 15, 20, 25, 30, 35]
  num_graph_cols = 3
  locs = gather_shots(hole_number, round_numbers=round_numbers, shot_number=shot_number)
  mean_shift_process(locs, bandwidths)

