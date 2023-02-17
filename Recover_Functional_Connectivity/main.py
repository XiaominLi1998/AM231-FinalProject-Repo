from dataclasses import dataclass

from inverse_covariance import QuicGraphicalLassoCV, QuicGraphicalLassoEBIC

import numpy as np
import scipy.io
from sklearn.covariance import graphical_lasso
from typing import List

from data_loader import RecordingData, SpontaneousRecordingData
from data_sampler import MissingTraceSampler, PartialSpontRecordingData
from recover_fconnectivity import FunctionalConnectivityRecoverer
import globals as glb
import utils
import evals


@dataclass
class SingleRunConfig:
  recording_subset: RecordingData
  missing_fsp: np.ndarray
  auxi_mats: np.ndarray
  psd_corrector: str = glb.ALT_PROJ
  glasso_alpha: float = None
  Kfolds: int = None
  gamma: float = 0
  full_sigma: np.ndarray = None
  full_gm_sigma: np.ndarray = None
  full_gm_prec: np.ndarray = None
  full_gm_lam: float = None
  rec_model: str = glb.LINEAR_REG
  show_eigvals: bool = False
  edge_types: int = 2
  edge_cutoff_tol: float = 1e-8
  edge_cutoff_qt: float = 0.05


@dataclass
class FCRecoveryConfig:
  sessions: list
  time_range: tuple
  sim_timestamps: int
  loaded_sessions: List[RecordingData] = None
  auxi_mask: list = None
  layersOI: np.ndarray = np.arange(4)
  xrange: tuple = (0.1, 0.3)
  yrange: tuple = (0.1, 0.3)
  ratio: bool = True
  init_useCorr: bool = True
  sim_layer_size: int = 1
  layer_stride: int = 1
  psd_corrector: str = glb.ALT_PROJ
  full_glasso_alpha: float = 0.1
  full_Kfolds: int = None
  full_gamma: float = None
  glasso_alpha: float = 0.1
  Kfolds: int = None
  gamma: float = None
  rec_model: str = glb.LINEAR_REG
  edge_types: int = 2
  edge_cutoff_tol: float = 1e-8
  edge_cutoff_qt: float = 0.05


def get_spont_auxi_mats(session_id, recording_data, missing_trace, verbose=False):
  """
  Returns a stack of pairwise auxiliary matrices based on neuron distance, run speed and pupil area
  """
  # Inter-neuron distance
  neuron_dist = np.array(recording_data.inter_neuron_distance(plot=False))

  # Note: 1. For tuning curve below, better not to specify binrange, since it varies across different datasets
  #       2. In some unlucky cases, some bins never captures neuron pairs without simultaneous observations

  # Run Speed
  runspeed_bins, nonempty_runspeed_bins, runspeed_tc = utils.tuning_curve(missing_trace, recording_data.run_speed,
                                                                          bins=33, verbose=False) # binrange=(0, 48)
  runspeed_tc_corr = utils.tc_mat_corr(runspeed_tc)


  speed_delta_bins, nonempty_speed_delta_bins, speed_delta_tc = utils.tuning_curve(np.diff(missing_trace, axis=1),
                                                        np.diff(recording_data.run_speed, axis=0),
                                                        bins=25, verbose=False) # binrange=(0, 36),
  speed_delta_tc_corr = utils.tc_mat_corr(speed_delta_tc)


  # Pupil Area
  ppArea_bins, nonempty_ppArea_bins, ppArea_tc = utils.tuning_curve(missing_trace, recording_data.pupil_area,
                                                                    bins=51, verbose=False) # binrange=(0, 7000),
  ppArea_tc_corr = utils.tc_mat_corr(ppArea_tc)


  ppArea_delta_bins, nonempty_ppArea_delta_bins, ppArea_delta_tc = utils.tuning_curve(np.diff(missing_trace, axis=1),
                                                          np.diff(recording_data.pupil_area, axis=0),
                                                          bins=51, verbose=False) # binrange=(0, 7000)
  ppArea_delta_tc_corr = utils.tc_mat_corr(ppArea_delta_tc)

  if verbose:
    print("running speed:", runspeed_tc.shape)
    print("neuron dist", neuron_dist.shape)
    print("running speed tc_sim", runspeed_tc_corr.shape)
    print("running speed change", speed_delta_tc_corr.shape)
    print("pupil area", ppArea_tc_corr.shape)
    print("pupil area change", ppArea_delta_tc_corr.shape)

  auxi_mats = np.stack([neuron_dist, ppArea_tc_corr, ppArea_delta_tc_corr, runspeed_tc_corr, speed_delta_tc_corr],
                       axis=0)

  print("Session %d: Auxiliary Matrix Created!" % session_id)
  return auxi_mats


def single_run(cfg: SingleRunConfig):
  """
  Recovers FC from missing_fsp with specified configurations

  :param cfg: A SingleRunConfig object
  :return: a SingleRunResult object containing full and estimated cov, prec, adj mats.
  """
  # Note: Default to correlation matrix
  full_sigma = cfg.recording_subset.sample_covariance_matrix(corr=True) if cfg.full_sigma is None else cfg.full_sigma

  # Glasso estimation from full sample covariance/correlation
  full_gm_sigma, full_gm_prec, full_gm_lam = cfg.full_gm_sigma, cfg.full_gm_prec, cfg.full_gm_lam
  if (full_gm_sigma is None) or (full_gm_prec is None):
    quic = QuicGraphicalLassoCV(cv=cfg.Kfolds)
    quic.fit(cfg.full_sigma)
    full_gm_sigma, full_gm_prec, full_gm_lam = quic.covariance_, quic.precision_, quic.lam_

  # Recover correlation and apply Glasso to get precision
  fc_rec = FunctionalConnectivityRecoverer(model=cfg.rec_model)

  if cfg.glasso_alpha is not None:
    fcRec_result = fc_rec.recover_func_conn(data_matrix=cfg.missing_fsp, auxi_info=cfg.auxi_mats, glasso_alpha=cfg.glasso_alpha,
                                            isCov=False, psd_corrector=cfg.psd_corrector, show_eigvals=cfg.show_eigvals)
  elif cfg.Kfolds is not None:
    fcRec_result = fc_rec.recover_func_conn_CV(data_mat=cfg.missing_fsp, auxi_info=cfg.auxi_mats, Kfolds=cfg.Kfolds,
                                               isCov=False, psd_corrector=cfg.psd_corrector, show_eigvals=cfg.show_eigvals)
  else:
    fcRec_result = fc_rec.recover_func_conn_EBIC(data_mat=cfg.missing_fsp, auxi_info=cfg.auxi_mats,gamma=cfg.gamma,
                                                 isCov=False, psd_corrector=cfg.psd_corrector, show_eigvals=cfg.show_eigvals)
  result = evals.SingleRunResult(fcRec_result, full_sigma, full_gm_sigma, full_gm_prec, full_gm_lam,
                                 cfg.edge_types, cfg.edge_cutoff_tol, cfg.edge_cutoff_qt)
  return result


# Consecutive layers within simultaneous neuron blocks
# - can try different overlap ratio
# Interleaving layers within simultaneous neuron blocks
def diff_session_diff_missing_by_layers(cfg: FCRecoveryConfig, consecutive=True):
  """
  Experiment script for recover FC from observations missing by layers.

  :param cfg: a FCRecoveryConfig object specifying the setup params of the experiment
  :param consecutive: True if missingness is by consecutive layer, otherwise assume interleaving layers
  :return: A mapping from session_id to full, missing data and results
  """
  spont_mats = scipy.io.loadmat(glb.DATA_DIR + "dbspont.mat").get("db")[0]
  print("Total: %d datasets" % (len(spont_mats)))

  session2res = {}
  missing_sampler = MissingTraceSampler()

  for session_i in cfg.sessions:
    if cfg.loaded_sessions is None:
      spont_fp = glb.DATA_DIR + "_".join(
        ["spont", spont_mats[session_i][glb.SESSION_NAME][0], spont_mats[session_i][glb.REC_DATE][0]])
      spont_dataset = SpontaneousRecordingData(data_fp=spont_fp)
    else:
      spont_dataset = cfg.loaded_sessions[cfg.sessions.index(session_i)]
    print("\nSession %d loaded!" % session_i, spont_dataset.session_name)

    start_tpt, end_tpt = cfg.time_range
    spont_subset = PartialSpontRecordingData(spont_dataset,layers=cfg.layersOI, x_range=cfg.xrange, y_range=cfg.yrange,
                                             start_timepoint=start_tpt, end_timepoint=end_tpt, ratio=cfg.ratio)
    print("Start, end time: ", start_tpt, end_tpt)
    print("Number of neurons: ", spont_subset.neuron_counts)

    # Estimate cov, prec on Full data with glasso
    full_sigma = spont_subset.sample_covariance_matrix(corr=cfg.init_useCorr, check_psd=True, display=False)
    if cfg.full_glasso_alpha is not None:
      full_gm_cov, full_gm_prec = graphical_lasso(full_sigma, alpha=cfg.glasso_alpha)
      full_gm_lam = cfg.glasso_alpha
    else:
      if cfg.full_Kfolds is not None:
        quic_full = QuicGraphicalLassoCV(cv=cfg.full_Kfolds)
      else:
        quic_full = QuicGraphicalLassoEBIC(gamma=cfg.full_gamma)
      quic_full.fit(np.transpose(spont_subset.fsp))  # X: n_samples, n_features
      full_gm_cov, full_gm_prec = quic_full.covariance_, quic_full.precision_
      full_gm_lam = quic_full.lam_

    # Simulate missingness
    if consecutive:
      missing_fsp = missing_sampler.missing_fsp_consecutive_layers(spont_subset, cfg.sim_timestamps,
                                                                   cfg.sim_layer_size, cfg.layer_stride)
    else:  # interleaving sim neuron blocks
      missing_fsp = missing_sampler.missing_fsp_interleaving_layers(spont_subset, cfg.sim_timestamps, cfg.sim_layer_size)

    # Obtain pairwise similarity matrices of all auxiliary information
    auxi_mats = get_spont_auxi_mats(session_i, spont_subset, missing_fsp)
    auxi_mats_to_use = auxi_mats[cfg.auxi_mask] if cfg.auxi_mask is not None else auxi_mats

    # build single run config
    singlerun_cfg = SingleRunConfig(spont_subset, missing_fsp, auxi_mats_to_use, cfg.psd_corrector,
                                    cfg.glasso_alpha, cfg.Kfolds, cfg.gamma,
                                    full_sigma, full_gm_cov, full_gm_prec, full_gm_lam,
                                    cfg.rec_model, False,
                                    cfg.edge_types, cfg.edge_cutoff_tol, cfg.edge_cutoff_qt)
    # Recover prec under missing fsp
    result = single_run(singlerun_cfg)

    # Save the partial dataset and result
    session2res[spont_dataset.session_name] = {"spont_subset": spont_subset,
                                               "missing_fsp": missing_fsp,
                                               "number of neurons": missing_fsp.shape[0],
                                               "result": result}
    print("Session %d Auxiliary Info Comparison Finished!\n" % session_i)
  return session2res


def diff_session_diff_auxi(cfg: FCRecoveryConfig, auxi_masks, consecutive=True):
  """
  Experiment script for recover FC for recordings missing by layers, using different auxiliary info as predictors.

  :param cfg: a FCRecoveryConfig object specifying the setup params of the experiment
  :param auxi_masks: A list of binary lists that indicates which auxi variables to include in the predictive model
  :param consecutive: True if missingness is by consecutive layer, otherwise assume interleaving layers
  :return: A mapping from session_id to full, missing data and results
  """
  spont_mats = scipy.io.loadmat(glb.DATA_DIR + "dbspont.mat").get("db")[0]
  print("%d datasets to loaded" % (len(spont_mats)))

  session2res = {}
  missing_sampler = MissingTraceSampler()

  for session_i in cfg.sessions:
    spont_fp = glb.DATA_DIR + "_".join(
      ["spont", spont_mats[session_i][glb.SESSION_NAME][0], spont_mats[session_i][glb.REC_DATE][0]])
    spont_dataset = SpontaneousRecordingData(data_fp=spont_fp)
    print("\nSession %d loaded!" % session_i, spont_dataset.session_name)

    start_tpt, end_tpt = cfg.time_range
    print("start, end time: ", start_tpt, end_tpt)

    spont_subset = PartialSpontRecordingData(spont_dataset,layers=cfg.layersOI, x_range=cfg.xrange, y_range=cfg.yrange,
                                             start_timepoint=start_tpt, end_timepoint=end_tpt, ratio=cfg.ratio)

    # Estimate cov, prec on Full data with glasso
    full_sigma = spont_subset.sample_covariance_matrix(corr=cfg.init_useCorr, check_psd=True, display=False)
    if cfg.full_glasso_alpha is not None:
      full_gm_cov, full_gm_prec = graphical_lasso(full_sigma, alpha=cfg.glasso_alpha)
      full_gm_lam = cfg.glasso_alpha
    else:
      if cfg.full_Kfolds is not None:
        quic_full = QuicGraphicalLassoCV(cv=cfg.full_Kfolds)
      else:
        quic_full = QuicGraphicalLassoEBIC(gamma=cfg.full_gamma)
      quic_full.fit(np.transpose(spont_subset.fsp))  # X: n_samples, n_features
      full_gm_cov, full_gm_prec = quic_full.covariance_, quic_full.precision_
      full_gm_lam = quic_full.lam_

    # Simulate missingness
    if consecutive:
      missing_fsp = missing_sampler.missing_fsp_consecutive_layers(spont_subset, cfg.sim_timestamps,
                                                                   cfg.sim_layer_size, cfg.layer_stride)
    else:  # interleaving sim neuron blocks
      missing_fsp = missing_sampler.missing_fsp_interleaving_layers(spont_subset, cfg.sim_timestamps, cfg.sim_layer_size)

    # Obtain pairwise similarity matrices of all auxiliary information
    auxi_mats = get_spont_auxi_mats(session_i, spont_subset, missing_fsp)

    results = []
    for auxi_mask in auxi_masks:
      auxi_mats_to_use = auxi_mats[np.where(np.array(auxi_mask) == 1)[0]]
      print("Auxi_mats_to_use: ", auxi_mats_to_use.shape)

      # build single run config
      singlerun_cfg = SingleRunConfig(spont_subset, missing_fsp, auxi_mats_to_use, cfg.psd_corrector,
                                      cfg.glasso_alpha, cfg.Kfolds, cfg.gamma,
                                      full_sigma, full_gm_cov, full_gm_prec, full_gm_lam,
                                      cfg.rec_model, False,
                                      cfg.edge_types, cfg.edge_cutoff_tol, cfg.edge_cutoff_qt)
      # Recover prec under missing fsp
      result = single_run(singlerun_cfg)
      results.append(result)

    # Save the partial dataset and result
    session2res[spont_dataset.session_name] = {"spont_subset": spont_subset,
                                               "missing_fsp": missing_fsp,
                                               "number of neurons": missing_fsp.shape[0],
                                               "result": results}
    print("Session %d Auxiliary Info Comparison Finished!\n" % session_i)
  return session2res




# TODO: do i need this later?
# def diff_mouse_diff_alphas(sessions, glasso_alphas, time_range, xrange=(0, 0.25), yrange=(0, 0.25), ratio=True,
#                            nblk_pct=0.2, tblk_pct=0.1, overlap_pct=0.5, auxi_mask=None, psd_corrector=glb.ALT_PROJ,
#                            verbose=False):
#   spont_mats = scipy.io.loadmat(glb.DATA_DIR + "dbspont.mat").get("db")[0]  # 9 datasets of recordings
#   print("%d datasets to loaded" % (len(spont_mats)))
#
#   session2res = {}
#
#   for session_i in sessions:
#     spont_fp = glb.DATA_DIR + "_".join(
#       ["spont", spont_mats[session_i][glb.SESSION_NAME][0], spont_mats[session_i][glb.REC_DATE][0]])
#     spont_dataset = SpontaneousRecordingData(data_fp=spont_fp)
#     print("\nSession %d loaded!" % session_i)
#
#     layersOI = range(0, spont_dataset.layer_counts, 2)
#     start_tpt = max(1, int(time_range[0] * spont_dataset.timestamp_counts))
#     end_tpt = int(min(1, time_range[1]) * spont_dataset.timestamp_counts)
#
#     spont_subset = PartialSpaceRecordingDataset(PartialTimeRecordingDataset(PartialLayerRecordingDataset(
#       spont_dataset, layersOI=layersOI),
#       start_timepoint=start_tpt, end_timepoint=end_tpt),
#       x_range=xrange, y_range=yrange, ratio=ratio)
#
#     # True sample correlation
#     full_sigma = spont_subset.sample_covariance_matrix(corr=True, check_psd=True, display=False)
#
#     # Sample Missing Blocks
#     neuronblk_size = int(spont_subset.neuron_counts * nblk_pct)
#     timeblk_size = int(spont_subset.timestamp_counts * tblk_pct)
#     sampler = MissingTraceSampler()
#     missing_fsp = sampler.sample_missing_traces_by_blocks(dataset=spont_subset,
#                                                           neuronblock_size=neuronblk_size,
#                                                           timeblock_size=timeblk_size,
#                                                           overlap_percent=overlap_pct)
#     # All auxiliary matrices
#     auxi_mats = get_spont_auxi_mats(session_i, spont_subset, missing_fsp)
#
#     # Vary lambdas
#     result_seq = []
#     for glasso_alpha in glasso_alphas:
#       result = single_run(spont_subset, missing_fsp, auxi_mats, glasso_alpha, psd_corrector, auxi_mask=auxi_mask,
#                           full_sigma=full_sigma)
#       result_seq.append(result)
#
#     session2res[spont_dataset.session_name] = {"neuron_counts": spont_subset.neuron_counts,
#                                                "results": result_seq}
#     print("Session %d Overlap Ratio Comparison Finished!\n")
#   return session2res

