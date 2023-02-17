from collections import namedtuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support


FCRecoveryResult = namedtuple("FCRecoveryResult", ["auxi_sigma", "auxi_fitter", "gm_sigma", "gm_prec", "lam"])


class SingleRunResult(object):

  def __init__(self, fcRec_result: FCRecoveryResult, full_sigma, full_gm_sigma, full_gm_prec, full_gm_lam,
               edge_types, edge_cutoff_tol, edge_cutoff_qt):
    self.fcRec_result = fcRec_result
    self.full_gm_lam = full_gm_lam

    # Frob norm of matrix estimation error
    self.auxi_sigma_err = mat_rmse(fcRec_result.auxi_sigma, full_sigma)
    self.gm_sigma_err = mat_rmse(fcRec_result.gm_sigma, full_gm_sigma)
    self.gm_prec_err = mat_rmse(fcRec_result.gm_prec, full_gm_prec)

    # Matrix correlation
    self.auxi_sigma_corr = mat_corr(fcRec_result.auxi_sigma, full_sigma)
    self.gm_sigma_corr = mat_corr(fcRec_result.gm_sigma, full_gm_sigma)
    self.gm_prec_corr = mat_corr(fcRec_result.gm_prec, full_gm_prec)

    self.full_A = prec_to_adjmat(full_gm_prec, standardize=True, qt=edge_cutoff_qt,
                                 tol=edge_cutoff_tol, edge_types=edge_types)
    self.Ahat = prec_to_adjmat(fcRec_result.gm_prec, standardize=True, qt=edge_cutoff_qt,
                                 tol=edge_cutoff_tol, edge_types=edge_types)

    # Accuracy, recall, precision, F-1
    self.acc = adj_mat_acc(self.full_A, self.Ahat)

    self.recall, self.prec, self.f1, _ = adj_mat_recall_prec_f1(self.full_A, self.Ahat)

  def __str__(self):
    single_result_str = ""
    for a in dir(self):
      if not a.startswith("__") and not callable(getattr(self, a)):
        single_result_str += "{:20} {}\n".format(a, getattr(self, a))
    return single_result_str


def mat_rmse(X, Xhat):
  # Computes Frobenius norm of ||X - Xhat|| divided by p
  assert np.shape(X) == np.shape(Xhat), "Input matrices must have the same shape!"
  diff = np.abs(X - Xhat)
  return np.linalg.norm(diff, ord="fro") / np.shape(X)[0]


def mat_corr(A, B, standardize=True, square_symmetric=True):
  # Computes Pearson correlation between matrix A and B
  shapeA, shapeB = np.shape(A), np.shape(B)
  assert shapeA == shapeB, "Input matrices has different shapes: %s, %s" % (str(shapeA), str(shapeB))
  if square_symmetric:
    idxA, idxB = np.triu_indices(shapeA[0], 1), np.triu_indices(shapeB[0], 1)  # Exclude diagonal
    upperA, upperB = A[idxA], B[idxB]
    return pearsonr(upperA, upperB)[0]
  return pearsonr(np.matrix(A).flatten(), np.matrix(B).flatten())[0]


def standardize_prec(prec):
  p = len(prec)
  sqrt_diag = np.diag(prec) ** 0.5
  standardizer = np.reshape(sqrt_diag, (p, 1)) * np.reshape(sqrt_diag, (1, p))
  std_prec = prec / standardizer
  return std_prec


def prec_to_adjmat(prec, qt=0.05, tol=1e-8, standardize=True, edge_types=2):
  std_prec = standardize_prec(prec) if standardize else prec
  A = np.zeros_like(std_prec)
  P = np.abs(std_prec)
  if tol is None:
    tol = np.quantile(P[P != 0], qt)  # prec is flattened in the computation
  if edge_types == 3:
    A[std_prec > tol] = 1.0
    A[std_prec < -tol] = -1.0
  else:
    A[P > tol] = 1.0
  return A


def adj_mat_acc(A1, A2):
  # Exclude diagonal entries
  assert np.shape(A1) == np.shape(A2), "Input matrices must have the same shape!"
  p = np.shape(A1)[0]

  upper1, upper2 = A1[np.triu_indices(p, 1)], A2[np.triu_indices(p, 1)]
  diff = np.abs(upper1 - upper2)
  return 1.0 - np.mean(diff)


def adj_mat_recall_prec_f1(A, Ahat):
  # Exclude diagonal entries
  assert np.shape(A) == np.shape(Ahat), "Input matrices must have the same shape!"
  p = np.shape(A)[0]

  upper1, upper2 = A[np.triu_indices(p, 1)], Ahat[np.triu_indices(p, 1)]
  return precision_recall_fscore_support(upper1.flatten(), upper2.flatten(), average="binary")


def adj_acc(A, Ahat):
  # Computes the percentage of matching entries of a binary adjacency matrix
  diff = np.abs(A - Ahat)
  return 1.0 - np.sum(diff) / np.size(A)
