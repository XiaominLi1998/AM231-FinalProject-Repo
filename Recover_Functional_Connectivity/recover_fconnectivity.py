import numpy as np
import pandas as pd
from inverse_covariance import quic, QuicGraphicalLassoCV, QuicGraphicalLassoEBIC
from inverse_covariance.adaptive_graph_lasso import AdaptiveGraphicalLasso

from sklearn.covariance import graphical_lasso
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.linear_model import LinearRegression

from NearestPSD import shrinking, nearest_correlation

from data_loader import RecordingData
from evals import FCRecoveryResult
import globals as glb
import utils
import plot_utils


# TODO: Neighborhood selection (modify existing code)

class FunctionalConnectivityRecoverer(object):

  def __init__(self, pd_maker="", glasso="", model=glb.LINEAR_REG):
    self.pd_maker = pd_maker
    self.glasso = glasso
    self.model = model

  def recover_fconn_graph(self):
    # TODO: directly predict edge connections rather than cov
    pass

  def recover_cov(self, data_matrix, predictors, isCov=True, display_fit=False):
    """
    This function recovers the precision matrix given a data matrix with missing blocks

    :param data_matrix: Calcium imaging fluorescence trace with nulls (neuron_counts, timepoints)
    :param predictors: Auxiliary information matrix (predictor_types, neuron_counts, neuron_counts)
    :param isCov: True if input sample covariance to glasso; if False, use correlation matrix instead
    :param display_fit: Show predictive model fit (can only support single predictor case)

    :return: An estimated precision matrix that represents the functional connectivity of each neuron pair
    """
    # Compute an incomplete covariance matrix
    N, T = data_matrix.shape
    sample_covariance = pd.DataFrame(data_matrix.T).cov() if isCov else pd.DataFrame(data_matrix.T).corr()
    # print("data matrix shape: ", data_matrix.shape)
    # print("sample cov shape: ", sample_covariance.shape)
    # print("predictors: ", predictors)

    # Build predictor and prediction variables from non-NaN entries
    X, y = [], []
    for i in range(N):
      for j in range(i, N):
        if not np.isnan(sample_covariance[i][j]) and not np.any(np.isnan(predictors[:, i, j])):
          X.append(predictors[:, i, j])
          y.append(sample_covariance[i][j])
    y = np.array(y)
    X = np.array(X)  # n_samples, n_features
    # print("predictor X shape: ", X.shape)
    if X.ndim == 1:
      X = X.reshape(-1, 1)

    # 'Regress' real-valued covariance on auxiliary infos (later can be expanded to an ensemble model)
    if self.model == glb.LINEAR_REG:
      model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True).fit(X, y)
      if display_fit and X.ndim == 1:
        plot_utils.plot_fitted_lr(X, y, model, isCov)

    # TODO: Implement the following
    elif self.model == glb.RIDGE_REG:

      pass

    elif self.model == glb.KERNEL_REG:

      pass

    elif self.model == glb.KERNEL_RIDGE:

      pass


    # Complete the covariance matrix with the fitted predictor
    for i in range(N):
      for j in range(i+1, N):
        if np.isnan(sample_covariance[i][j]):
          xvec = predictors[:, i, j].reshape(1, -1)
          sample_covariance[i][j] = sample_covariance[j][i] = model.predict(xvec)

    print("Final Correlation matrix contains NaN?", np.any(np.isnan(sample_covariance)))

    return sample_covariance, model


  def make_psd(self, cov, method="newton", show_eigen_vals=False):
    """
    Given an estimated covariance matrix, perturb it to a positive semi-definite matrix if it's not so
    :param cov:
    :return:
    """
    # Check for PDness, if not, perturb it to the closest PD matrix
    # if not shrinking.checkPD(sample_covariance, exception=False):
    cov_psd = np.zeros_like(cov)
    if not utils.isPSD(cov):
      print("Estimated sample covariance is not PSD, fixing with %s method..." % method)
      if method == glb.NEWTON:
        print("M0:", cov.shape)
        print("M1:", np.identity(len(cov)).shape)
        cov_psd = utils.newtons_PDCorrection(cov, M1=np.identity(len(cov)))
      elif method == glb.ALT_PROJ:
        cov_psd = nearest_correlation.nearcorr(cov, max_iterations=500)

      if show_eigen_vals:
        plot_utils.display_eigenvals(cov, value_range=(float("-inf"), float("inf")))
        plot_utils.display_eigenvals(cov, value_range=(float("-inf"), 0))
      return cov_psd

    return cov

  def complete_psd_cov(self, data_mat, auxi_info, isCov=False, psd_corrector="newton",
                 display_fit=False, show_eigvals=False):
    # Complete Covariance
    estCov, auxi_predictors = self.recover_cov(data_mat, auxi_info, isCov=isCov, display_fit=display_fit)
    # TODO: Standardization may not matter
    # TODO: Check if there's any predicted value outside [-1, 1]
    # TODO: Regress on the Fisher transformation to guarantee the vals falling between [-1, 1]
    estCov = self.make_psd(estCov, method=psd_corrector, show_eigen_vals=show_eigvals)

    return estCov, auxi_predictors


  # TODO: Add objective function type?
  def recover_func_conn_CV(self, data_mat, auxi_info, Kfolds=4, isCov=False, psd_corrector="newton", display_fit=False,
                           show_eigvals=False):
    # Default to 4-fold cross validation to select optimal lambda
    estCov, auxi_fitter = self.complete_psd_cov(data_mat, auxi_info, isCov, psd_corrector, display_fit, show_eigvals)

    quic = QuicGraphicalLassoCV(cv=Kfolds)

    quic.fit(estCov)  # TODO: Wrong!! Input should be data_mat! (But with missing data, it's hard to use CV here.)
    return FCRecoveryResult(estCov, quic.covariance_, quic.precision_, auxi_fitter, quic.lam_)


  def recover_func_conn_EBIC(self, data_mat, auxi_info, gamma=0, isCov=False, psd_corrector="newton", display_fit=False,
                             show_eigvals=False):
    # Default to BIC selection criteria (gamma = 0)
    estCov, auxi_fitter = self.complete_psd_cov(data_mat, auxi_info, isCov, psd_corrector, display_fit, show_eigvals)

    quic_ebic = QuicGraphicalLassoEBIC(gamma=gamma)
    quic_ebic.fit(estCov)
    return FCRecoveryResult(estCov, auxi_fitter, quic_ebic.covariance_, quic_ebic.precision_, quic_ebic.lam_)


  def recover_func_conn(self, data_matrix, auxi_info, glasso_alpha, isCov=False, psd_corrector="newton",
                 display_fit=False, show_eigvals=False):
    # Complete Covariance
    estCov, reg = self.complete_psd_cov(data_matrix, auxi_info, isCov, psd_corrector, display_fit, show_eigvals)

    # Estimate functional connectivity via Glasso with specified regularization strength
    glasso_corr, glasso_prec = graphical_lasso(estCov, alpha=glasso_alpha)
    return FCRecoveryResult(estCov, reg, glasso_corr, glasso_prec, glasso_alpha)


  def display_config(self):
    """
    Display Configuration values.
    """
    print("\nFCRecoverer Configurations:")
    for a in dir(self):
      if not a.startswith("__") and not callable(getattr(self, a)):
        print("{:30} {}\n".format(a, getattr(self, a)))
    print("\n")

