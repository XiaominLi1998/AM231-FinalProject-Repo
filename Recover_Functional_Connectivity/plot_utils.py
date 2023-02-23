import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx

from evals import standardize_prec
from typing import List
from evals import SingleRunResult


def display_missing_fsp(missing_fsp):
  X = np.nan_to_num(missing_fsp, nan=-9999)  # set nan entries to -9999
  ax = sns.heatmap(X, cmap="plasma")
  ax.set_xlabel("Timepoints")
  ax.set_ylabel("Neurons")
  ax.set_title("Neuron Trace with Missing Entries")
  plt.show()


def display_missing_cov(missing_cov, cmap="viridis"):
  X = np.nan_to_num(missing_cov, nan=-9999)
  ax = sns.heatmap(X, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  ax.set_title("Covariance of a Neuron Trace with Missing Entries")
  plt.show()


def display_eigenvals(X, value_range=(float("-inf"), float("inf"))):
  low, high = value_range[0], value_range[1]
  eigvals = np.linalg.eigvals(X)
  plot_vals = eigvals[(low < eigvals) & (eigvals <= high)]

  # Plot the eigenvalue distribution
  plt.hist(plot_vals, bins=50, color="purple", alpha=0.75)
  plt.title("Eigenvalue Distribution")
  plt.xlabel("Eigenvalues")
  plt.ylabel("Number of eigenvalues")
  plt.show()


# Plot a pair-wise matrix via heatmap
def plot_matrix(mat, title="", cmap="BuPu", figsize=(10, 7)):
  fig, ax = plt.subplots(figsize=figsize)
  if not title:
    title = "Matrix of %d neurons" % mat.shape[0]
  ax.set_title(title, fontsize=16)
  sns.heatmap(mat, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  plt.show()


# Plot precision matrix
def plot_prec(prec, alpha, ax=None, standardize=True, label="", cmap="viridis"):
  P = np.array(prec)
  if standardize:
    P = standardize_prec(prec)
  if ax:
    sns.heatmap(P, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(P, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  ax.set_title(r"Precision Matrix [%s, $\lambda$ = %.2f]" % (label, alpha))
  plt.show()


# Plot adjacency matrix
def plot_adj_mat(A, ax=None, label="", include_negs=False):
  plt.figure(figsize=(12, 9))
  A2 = A * 3000
  cmap = "cividis" if not include_negs else "bwr"
  if ax:
    sns.heatmap(A2, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(A2, cmap=cmap)
  ax.set_xlabel("Neurons", fontsize=18)
  ax.set_ylabel("Neurons", fontsize=18)
  total_edges = len(A[A != 0])
  if not include_negs:
    ax.set_title("Adjacency Matrix [%s] [%d edges]" % (label, total_edges), fontsize=20, y=1.05)
  else:
    pos_edges = len(A[A > 0])
    neg_edges = len(A[A < 0])
    ax.set_title("Adjacency Matrix [%s] [%d edges: %d+, %d-]" % (label, total_edges, pos_edges, neg_edges), y=1.05)
  plt.show()


# Plot connectivity graph based on an adjacency matrix and a location matrix
def plot_connectivity_graph(A, xys, cmap=plt.cm.coolwarm, ax=None, label=""):
  plt.figure(figsize=(12, 9))

  G = nx.convert_matrix.from_numpy_array(A)
  colors = [A[i][j] for i, j in G.edges]
  nx.draw_networkx(G, pos=xys, node_color="orange", alpha=0.85,
                   width=2, edge_cmap=cmap, edge_color=colors)
  plt.axis('equal')


def plot_fitted_lr(X, y, reg, cov):
  plt.scatter(X, y, color="red", alpha=0.7)
  plt.plot(X, reg.predict(X), color="blue")
  plt.title("Fitted Linear Regression")
  plt.xlabel("Auxiliary Information")
  plt.ylabel("Covariance") if cov else plt.ylabel("Correlation")
  plt.show()


def unpack_result_list(results: List[SingleRunResult]):
  lams = []
  auxi_rmse, gm_sigma_rmse, gm_prec_rmse = [], [], []
  auxi_corr, gm_sigma_corr, gm_prec_corr = [], [], []
  acc, prec, recall, f1 = [], [], [], []
  adj_mats = []
  for result in results:
    lams.append(result.fcRec_result.lam)
    auxi_rmse.append(result.auxi_sigma_err)
    auxi_corr.append(result.auxi_sigma_corr)
    gm_sigma_rmse.append(result.gm_sigma_err)
    gm_sigma_corr.append(result.gm_sigma_corr)
    gm_prec_rmse.append(result.gm_prec_err)
    gm_prec_corr.append(result.gm_prec_corr)
    acc.append(result.acc)
    prec.append(result.prec)
    recall.append(result.recall)
    f1.append(result.f1)
    adj_mats.append(result.Ahat)
  return lams, auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
         acc, prec, recall, f1, adj_mats


# TODO: Add function to display multiple session results in the same plot


def display_results_diff_auxi(auxi_labels, session2res):

  indx = range(len(auxi_labels))
  my_colors = ["grey", "purple", "blue", "orange", "green", "red"]

  for session, res in session2res.items():
    neuron_cnts = res.get("number of neurons")
    label = session + r"$ (%d neurons)$" % (neuron_cnts)
    results = res.get("result")
    lams, auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
      acc, prec, recall, f1, adj_mats = unpack_result_list(results)

    # Output gm estimation for full data
    opt_lam_full = res.get("full_gm_lam")
    print("Optimal Lambda for full data: ", opt_lam_full)
    plot_adj_mat(A=results[0].full_A, label=label + "--full data")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(22, 16))
    bar1 = ax1.bar(indx, auxi_rmse, alpha=0.8, color=my_colors, width=0.5, label=label)
    ax1.set_title(r"$||S - \hat{\Sigma}||_F / p$", fontweight="bold")
    ax1.set_xticks(indx)
    ax1.set_xticklabels(auxi_labels, rotation=20)
    ax1.set_ylabel("Frobenius norm")
    ax1.set_ylim([0, max(auxi_rmse) * 1.2])
    autolabel(bar1, [round(x, 3) for x in auxi_rmse], ax1)

    bar2 = ax2.bar(indx, auxi_corr, alpha=0.6, color=my_colors, width=0.5, label=label)
    ax2.set_title("Correlation of $S, \hat{\Sigma}$", fontweight="bold")
    ax2.set_xticks(indx)
    ax2.set_xticklabels(auxi_labels, rotation=20)
    ax2.set_ylabel("Matrix correlation")
    ax2.set_ylim([0, max(auxi_corr) * 1.2])
    autolabel(bar2, [round(x, 3) for x in auxi_corr], ax2)

    #ax3.bar(indx, gm_sigma_mes, alpha=0.8, color=my_colors, width=0.5, label=label)
    bar3 = ax3.bar(indx, gm_sigma_rmse, alpha=0.8, color=my_colors, width=0.5, label=label)
    ax3.set_title(r"$||\hat{\Sigma}_{GM(S, \lambda)} - \hat{\Sigma}_{GM(\hat{\Sigma}, \lambda)}||_F / p$",
                  fontweight="bold")
    ax3.set_xticks(indx)
    ax3.set_xticklabels(auxi_labels, rotation=20)
    ax3.set_ylabel("Frobenius norm")
    ax3.set_ylim([0, max(gm_sigma_rmse) * 1.2])
    autolabel(bar3, [round(x, 3) for x in gm_sigma_rmse], ax3)

    bar4 = ax4.bar(indx, gm_sigma_corr, alpha=0.6, color=my_colors, width=0.5, label=label)
    ax4.set_title("Correlation of $\hat{\Sigma}_{GM(S, \lambda)} - \hat{\Sigma}_{GM(\hat{\Sigma}, \lambda)}$",
                  fontweight="bold")
    ax4.set_xticks(indx)
    ax4.set_xticklabels(auxi_labels, rotation=20)
    ax4.set_ylabel("Matrix correlation")
    ax4.set_ylim([0, max(gm_sigma_corr) * 1.2])
    autolabel(bar4, [round(x, 3) for x in gm_sigma_corr], ax4)

    bar5 = ax5.bar(indx, gm_prec_rmse, alpha=0.8, color=my_colors, width=0.5, label=label)
    ax5.set_title(r"$||\hat{\Theta}_{S, \lambda} - \hat{\Theta}_{\lambda}||_F / p$", fontweight="bold")
    ax5.set_xticks(indx)
    ax5.set_xticklabels(auxi_labels, rotation=20)
    ax5.set_ylabel("Frobenius norm")
    ax5.set_ylim([0, max(gm_prec_rmse) * 1.2])
    autolabel(bar5, [round(x, 3) for x in gm_prec_rmse], ax5)

    bar6 = ax6.bar(indx, gm_prec_corr, alpha=0.6, color=my_colors, width=0.5, label=label)
    ax6.set_title(r"Correlation of $\hat{\Theta}_{S, \lambda}, \hat{\Theta}_{\lambda}$", fontweight="bold")
    ax6.set_xticks(indx)
    ax6.set_xticklabels(auxi_labels, rotation=20)
    ax6.set_ylabel("Matrix correlation")
    ax6.set_ylim([0, max(gm_prec_corr) * 1.2])
    autolabel(bar6, [round(x, 3) for x in gm_prec_corr], ax6)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

    # Plot accuracy, precision, recall and F1
    plot_adj_evals_diff_auxi(label, auxi_labels, acc, prec, recall, f1)

    # Plot adjacency matrices with different combo of auxiliary variables
    for i in range(len(adj_mats)):
      plot_adj_mat(A=adj_mats[i], label=label + "" + auxi_labels[i], include_negs=False)


def autolabel(bar_plot, bar_label, ax):
  for idx, rect in enumerate(bar_plot):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
            bar_label[idx],
            ha='center', va='bottom', rotation=0)


def plot_adj_evals_diff_auxi(label, auxi_labels, acc, prec, recall, f1):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
  indx = range(len(auxi_labels))
  my_colors = ["grey", "purple", "blue", "orange", "green", "red"]

  bar1 = ax1.bar(indx, acc, alpha=0.7, color=my_colors, width=0.5, label=label)
  ax1.set_title("Accuracy of Recovered Adjacency Matrix", fontweight="bold")
  ax1.set_xticks(indx)
  ax1.set_xticklabels(auxi_labels, rotation=20)
  ax1.set_ylabel("Accuracy")
  ax1.set_ylim([0, 1.1])
  autolabel(bar1, [round(x, 3) for x in acc], ax1)

  bar2 = ax2.bar(indx, prec, alpha=0.7, color=my_colors, width=0.5, label=label)
  ax2.set_title("Precision of Recovered Adjacency Matrix", fontweight="bold")
  ax2.set_xticks(indx)
  ax2.set_xticklabels(auxi_labels, rotation=20)
  ax2.set_ylabel("Precision")
  ax2.set_ylim([0, max(prec) * 1.1])
  autolabel(bar2, [round(x, 3) for x in prec], ax2)


  bar3 = ax3.bar(indx, recall, alpha=0.7, color=my_colors, width=0.5, label=label)
  ax3.set_title("Recall of Recovered Adjacency Matrix", fontweight="bold")
  ax3.set_xticks(indx)
  ax3.set_xticklabels(auxi_labels, rotation=20)
  ax3.set_ylabel("Recall")
  ax3.set_ylim([0, max(recall) * 1.1])
  autolabel(bar3, [round(x, 3) for x in recall], ax3)

  bar4 = ax4.bar(indx, f1, alpha=0.7, color=my_colors, width=0.5, label=label)
  ax4.set_title("F-1 Score of Recovered Adjacency Matrix", fontweight="bold")
  ax4.set_xticks(indx)
  ax4.set_xticklabels(auxi_labels, rotation=20)
  ax4.set_ylabel("F-1")
  ax4.set_ylim([0, max(f1) * 1.1])
  autolabel(bar4, [round(x, 3) for x in f1], ax4)

  plt.subplots_adjust(hspace=0.3)
  plt.show()

# # Plot 3 precision matrices
# def plot_3_precs(p0, p1, p2, r0, r1, r2, ax=None, standardize_prec=True):
#
#   plot_prec(p0, r0, standardize=standardize_prec,
#             ax=ax, label="Full Sample")
#
#   plot_prec(p1, r1, standardize=standardize_prec,
#             ax=ax, label="RLA Uniform")
#
#   plot_prec(p2, r2, standardize=standardize_prec,
#             ax=ax, label="RLA Minvar")
#
#
# def plot_3_adj_mats(A0, A1, A2, ax=None, include_negs=False):
#
#   plot_adj_mat(A0, ax, label="Full Sample", include_negs=include_negs)
#   plot_adj_mat(A1, ax, label="RLA Uniform", include_negs=include_negs)
#   plot_adj_mat(A2, ax, label="RLA Minvar", include_negs=include_negs)
#
#
# def plot_metric_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
#   plt.figure(figsize=(12, 9))
#   metric_uni = [[] for _ in cs]
#   metric_minvar = [[] for _ in cs]
#   for sidx in session_ids:
#     for ci in range(len(cs)):
#       c = cs[ci]
#       metric_uni[ci].extend(metric_all[sidx]["uni"][c])
#       metric_minvar[ci].extend(metric_all[sidx]["minvar"][c])
#   metric_uni = [np.mean(f) for f in metric_uni]
#   metric_minvar = [np.mean(f) for f in metric_minvar]
#
#   plt.plot(cs, metric_uni, "o-", label="RLA uniform")
#   plt.plot(cs, metric_minvar, "o-", label="RLA minvar")
#   #plt.title("%s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
#   plt.xlabel("RLA Sampling Percentage", fontsize=18)
#   plt.ylabel(metric_name, fontsize=18)
#   plt.legend(prop={'size': 20})
#
#   if save_name:
#     plt.savefig(save_name)
#   plt.show()
#
#
# def plot_metric_uni_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
#   # assert cs is sorted ascendingly
#   plt.figure(figsize=(12, 9))
#   for sidx in session_ids:
#     metric = metric_all[sidx]
#     mean_metric_uni = [np.mean(metric["uni"][c]) for c in cs]
#     plt.plot(cs, mean_metric_uni, "o-", label="RLA uniform [Session %d]" % sidx)
#
#   #plt.title("Uniform RLA Sampling %s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
#   plt.xlabel("RLA Sampling Percentage", fontsize=18)
#   plt.ylabel(metric_name, fontsize=18)
#   plt.legend(prop={'size': 13})
#
#   if save_name:
#     plt.savefig(save_name)
#   plt.show()
#
#
# def plot_metric_minvar_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
#   # assert cs is sorted ascendingly
#   plt.figure(figsize=(12, 9))
#   for sidx in session_ids:
#     metric = metric_all[sidx]
#     mean_frobs_minvar = [np.mean(metric["minvar"][c]) for c in cs]
#     plt.plot(cs, mean_frobs_minvar, "o-", label="RLA min-var [Session %d]" % sidx)
#
#   #plt.title("Min-var RLA Sampling %s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
#   plt.xlabel("RLA Sampling Percentage", fontsize=18)
#   plt.ylabel(metric_name, fontsize=18)
#   plt.legend(prop={'size': 13})
#
#   if save_name:
#     plt.savefig(save_name)
#   plt.show()
#
#
# # Plot matrix correlation over different c
# def plot_corrs_over_cs(cs, corrs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, corrs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, corrs["minvar"], "o-", label="RLA minvar")
#   plt.title("Precision Matrix Correlation [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"Matrix Correlation", fontsize=18)
#   plt.legend(prop={'size': 13})
#
#   plt.show()
#
#
# # Plot edge estimation accuracy over different c
# def plot_acc_over_cs(cs, accs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, accs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, accs["minvar"], "o-", label="RLA minvar")
#   plt.title("Graph Structure Accuracy [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"Graph Structure Accuracy", fontsize=18)
#   plt.legend(prop={'size': 13})
#   plt.show()
#
#
# # Plot TPR over different c
# def plot_tpr_over_cs(cs, tprs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, tprs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, tprs["minvar"], "o-", label="RLA minvar")
#   plt.title("True Positive Rate of Edge Estimation [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"TPR", fontsize=18)
#   plt.legend(prop={'size': 13})
#   plt.show()
#
#
# # Plot FDR over different c
# def plot_fdr_over_cs(cs, fdrs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, fdrs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, fdrs["minvar"], "o-", label="RLA minvar")
#   plt.title("False Discovery Rate of Edge Estimation [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"FDR", fontsize=18)
#   plt.legend(prop={'size': 13})
#   plt.show()
#

