# Loads .mat datasets into RecordingData objects with methods to visualize its meta information

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.spatial.distance import pdist, squareform
from sklearn.covariance import EmpiricalCovariance
import seaborn as sns

import globals
from NearestPSD import shrinking
import utils


# Recording with 32 drifting gratings
# Keys:
#  - Fsp: a matrix of size number of neurons by number of timepoints of deconvolved fluorescence traces.
#  - stat: single cell statistics of cell detection algorithm (Suite2p)
#  - stim: properties of the stimulus
#  - med: estimated 3D position of cells in tissue.

class RecordingData(object):
  """
  This class is a container for calcium imaging data with auxiliary physiological and behavioral information.

  The cellular fluorescence traces in all layers of the brain are recorded simultaneously.
  Dataset with 10 layers are scanned at a rate of 3 Hz, those with 12 layers 2.5 Hz.
  """
  def __init__(self, recording_data=None, data_fp=None):
    if data_fp is not None:
      self.recording = scipy.io.loadmat(data_fp)

      # Neural signals
      self.fsp = np.array(self.recording.get("Fsp"))  # Neural responses * timepoints (remove t=0 with all zeros)
      self.neuron_counts = self.fsp.shape[0]
      zerovar_neurons = self.clean_fsp()
      self.timestamp_counts = self.fsp.shape[1]
      self.timestamp_offset = 0

      # Estimated 3D position of the cell
      self.neuron_xyz = np.array(self.recording.get("med"))
      if len(zerovar_neurons) > 0:
        self.neuron_xyz = np.delete(self.neuron_xyz, zerovar_neurons, axis=0)  # Neural responses * 3

      self.layers_zcoord = np.array(sorted(set(self.neuron_xyz[:, 2])))
      self.layer_counts = len(self.layers_zcoord)
      # Sort neuron index by layer
      self.layer2neurons = {zcoord: np.where(self.neuron_xyz[:, 2] == zcoord)[0] for zcoord in self.layers_zcoord}
      self.single_cell_stat = np.array(self.recording.get("stat")[0])  # [‘npix’]?

    else:
      # Copy the attributes from another RecordingData object
      # TODO: Figure out how to use it in jupyter notebook, maybe we don't need it.
      # Note: Jupyter notebook's autoreload cannot handle this check, disable it for now
      # assert isinstance(recording_data, RecordingData)
      # self.__dict__ = deepcopy(recording_data.__dict__)
      pass


  def clean_fsp(self):
    # Remove invalid fsp:
    # - nan, inf
    # - signal = 0 for all neurons
    # - signal = constant all the time

    nan_cols = np.where(np.isnan(self.fsp))[1]
    if len(nan_cols) > 0:
      print("Removing %d columns that contain NaN!" % len(nan_cols))
      self.fsp = np.delete(self.fsp, nan_cols, axis=1)

    inf_cols = np.where(np.isinf(self.fsp))[1]
    if len(inf_cols) > 0:
      print("Removing %d columns that contain Inf!" % len(inf_cols))
      self.fsp = np.delete(self.fsp, inf_cols, axis=1)

    # Skip the time points where all neural signals are 0
    allzero_cols = []
    p, T = self.fsp.shape
    for c in range(T):
      if np.all(self.fsp[:, c] == 0):
        allzero_cols.append(c)
    if len(allzero_cols) > 0:
      print("Removing %d all-zero columns" % len(allzero_cols))
      self.fsp = np.delete(self.fsp, allzero_cols, axis=1)

    # Check if any neuron has uniform signal (i.e. var = 0)
    zerovar_neurons = []
    for i in range(p):
      if np.var(self.fsp[i]) == 0:
        zerovar_neurons.append(i)
    if len(zerovar_neurons) > 0:
      print("Removing %d zero-variance neurons" % len(zerovar_neurons))
      self.fsp = np.delete(self.fsp, zerovar_neurons, axis=0)
    return zerovar_neurons

  def validate(self):
    # Validate the initialization
    pass

  def get_layers_fsp(self, layers=None):
    if layers is None:
      layers = list(range(self.layer_counts))
    fsp = []
    for layer in layers:
      fsp.extend(self.get_layer_fsp(layer))
    return np.array(fsp)

  def get_layer_fsp(self, layer):
    # return fsp of a specific layer
    if not (0 <= layer < self.layer_counts):
      print("[RecordingData] Invalid layer! There are %d neuron layers of this dataset!" % self.layer_counts)
    zcoord = self.layers_zcoord[layer]
    return np.array(self.fsp[self.layer2neurons[zcoord]])

  def get_layers_xyz(self, layers=None):
    # return fsp of specific layers
    if layers is None:
      layers = list(range(self.layer_counts))
    new_xyz = []
    for layer in layers:
      xyz = self.get_layer_xyz(layer)
      new_xyz.extend(xyz)
    return np.array(new_xyz)

  def get_layer_xyz(self, layer):
    # return xyz neuron coordinate of a specific layer
    if not (0 <= layer < self.layer_counts):
      print("[RecordingData] Invalid layer! There are %d neuron layers of this dataset!" % self.layer_counts)
    zcoord = self.layers_zcoord[layer]
    res = np.array(self.neuron_xyz[np.where(self.neuron_xyz[:, 2] == zcoord)[0]])
    return res

  # Visualizations
  def display_field_shapes(self):
    print("neuron counts:", self.neuron_counts)
    print("layer counts:", self.layer_counts)
    for layer in range(self.layer_counts):
      print("Layer %d: %d neurons" % (layer, len(self.layer2neurons[self.layers_zcoord[layer]])))
    print("timestamp counts", self.timestamp_counts)

  def display_fsp_distribution_all_time(self, bins=50):
    fsp_mean, fsp_std = np.mean(self.fsp, axis=0), np.std(self.fsp, axis=0)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax1.hist(fsp_mean, bins, color="green", alpha=0.8)
    ax1.set_title("Distribution of the Population Mean of the Calcium Imaging Signal")
    ax1.set_xlabel("Deconvolved Fluorescence Trace")
    ax1.set_ylabel("Number of Timepoints")

    ax2 = fig.add_subplot(122)
    ax2.hist(fsp_std, bins, color="green", alpha=0.6)
    ax2.set_title("Distribution of the Population Std of the Calcium Imaging Signal")
    ax2.set_xlabel("Deconvolved Fluorescence Trace")
    ax2.set_ylabel("Number of Timepoints")
    plt.show()

  def display_fsp_distribution_instant(self, timepoint=0, xrange=(0, 200), bins=100):
    assert timepoint < self.timestamp_counts, "Only %d timepoints are available in this recording!" % self.timestamp_counts

    instant_fsp = self.fsp[:, timepoint]
    plt.hist(instant_fsp, bins, color="blue", alpha=0.7)
    plt.xlim(xmin=xrange[0], xmax=xrange[1])
    plt.title("Snapshot Distribution of the Calcium Imaging Signal")
    plt.xlabel("Deconvolved Fluorescence Trace")
    plt.ylabel("Number of Neurons")
    plt.show()

  def display_rand_subset(self, size=5, use_seed=False):
    if use_seed: np.random.seed(globals.SEED) # TODO: Edit me!
    subset_fsp = self.fsp[np.random.choice(range(self.neuron_counts), size, replace=False)]
    for i in range(size):
      plt.plot(subset_fsp[i])
      title = "Calcium Imaging Value of Random Neuron %d" % i
      if isinstance(self, SpontaneousRecordingData):
        title += " [Session: %s]" % self.session_name
      plt.title(title, fontsize=16, fontweight='bold')
      plt.xlabel("Timestamp")
      plt.ylabel("Calcium Imaging Value")
      plt.show()

  def display_neuron_3d(self, session_name="", layers=None):
    ax = plt.axes(projection='3d')
    if layers is None:
      ax.scatter3D(self.neuron_xyz[:, 0], self.neuron_xyz[:, 1], self.neuron_xyz[:, 2],
                   c=self.neuron_xyz[:, 2], cmap="viridis", linewidth=0.5, s=28)
    else:
      zs = [self.layers_zcoord[layer] for layer in layers]
      neurons = [self.layer2neurons[z] for z in zs]
      for ns in neurons:
        ax.scatter3D(self.neuron_xyz[ns, 0], self.neuron_xyz[ns, 1], self.neuron_xyz[ns, 2],
                     c=self.neuron_xyz[ns, 2], cmap="viridis", linewidth=0.5, s=25)

    ax.set_title("Neuron 3D Locations", fontsize=20, fontweight="bold")
    ax.set_xlabel("x (um)", fontsize=16)
    ax.set_ylabel("y (um)", fontsize=16)
    ax.set_zlabel("Depth (um)", fontsize=16)

  def display_single_neuron_fsp(self, neuron_idx):
    ts = np.arange(self.timestamp_counts) / 180 if self.layer_counts <= 10 else np.arange(self.timestamp_counts) / 150
    plt.plot(ts, self.fsp[neuron_idx])
    plt.title("Neuron Fluorescence Trace", fontsize=17)
    plt.xlabel("Timestamp (min)", fontsize=14)
    plt.ylabel("Fluorescence Intensity", fontsize=14)
    plt.show()


  # Auxiliary Information
  def inter_neuron_distance(self, plot=False):
    # Inter-neuron distance (Euclidean)
    dist_condensed = pdist(self.neuron_xyz)
    if plot:
      # Plot the distribution of the inter-neuron distance
      plt.hist(dist_condensed, bins=30, color="grey")
      plt.title("Distribution of inter-neuron distance")
      plt.xlabel("Distance (um)")
      plt.ylabel("Count of Neuron Pairs")
      plt.show()
    return squareform(dist_condensed)

  def sample_covariance_matrix(self, corr=False, neuron_range=None, layer=None, check_psd=False, display=False, cmap="BuPu"):

    X, layer_name = (self.get_layer_fsp(layer), str(layer)) if layer is not None else (self.fsp, "all")
    X = X[neuron_range[0]:neuron_range[1]] if neuron_range else X
    cov, mat_name = (np.cov(X), "Covariance") if not corr else (np.corrcoef(X), "Correlation")

    if check_psd:
      is_pd = np.all(np.linalg.eigvals(cov) > 0)
      chol_is_pd = shrinking.checkPD(cov)
      if is_pd and chol_is_pd:
        print("Sample %s Matrix is positive definite!" % mat_name)
      elif is_pd and not chol_is_pd:
        print("Checker inconsistent with Cholesky decomposition checker!")
      else:
        is_psd = np.all(np.linalg.eigvals(cov) >= 0)
        print("Sample %s Matrix is positive semi-definite: %s" % (mat_name, is_psd))

    if display:
      fig, ax = plt.subplots(figsize=(10, 7))
      title = "Sample %s Matrix - Layer: %s" % (mat_name, layer_name)
      if isinstance(self, SpontaneousRecordingData):
        title += " [Session: %s]" % self.session_name
      ax.set_title(title)
      sns.heatmap(cov, cmap=cmap)
      ax.set_xlabel("Neurons")
      ax.set_ylabel("Neurons")
      plt.show()
    return cov

    #XM: added neuron_range parameter
  def sample_precision_matrix(self,  neuron_range=None, display=False):
    X = self.fsp 
    X = X[neuron_range[0]:neuron_range[1]] if neuron_range else X

    emp_cov_est = EmpiricalCovariance().fit(X.transpose())
    prec = emp_cov_est.precision_

    if display:
      title = "Empirical Correlation"
      if isinstance(self, SpontaneousRecordingData):
        title += " [Session: %s]" % self.session_name
      plt.title(title, fontsize=16, fontweight="bold")
      plt.imshow(prec, cmap=plt.cm.get_cmap("RdBu_r"))
      plt.xticks(())
      plt.yticks(())
    return prec


# Spontaneous Recording
class SpontaneousRecordingData(RecordingData):

  def __init__(self, recording_data=None, data_fp=None):
    super().__init__(recording_data, data_fp)
    if data_fp is not None:
      # Behavior data
      beh_data = self.recording.get("beh")
      self.run_speed = np.array(beh_data["runSpeed"][0][0][1:])
      self.motion_svd = np.array(beh_data[0]["face"][0]["motionSVD"][0][0][1:])
      self.motion_mask = np.array(beh_data[0]["face"][0]["motionMask"][0][0])
      self.avgframe = np.array(beh_data[0]["face"][0]["avgframe"][0][0])  # avg frame of face ROI
      self.pupil_area = np.array(beh_data[0]["pupil"][0]["area"][0][0][1:])
      self.pupil_com = np.array(beh_data[0]["pupil"][0]["com"][0][0][1:])  # pupil center of mass
      self.eye_motion_svd = np.array(beh_data[0]["eye"][0]["motionSVD"][0][0][1:])
      self.whisker_motion_svd = np.array(beh_data[0]["whisker"][0]["motionSVD"][0][0][1:])
      self.whisker_motion_mask = np.array(beh_data[0]["whisker"][0]["motionMask"][0][0])
      self.face_roi_shape = self.avgframe.shape

      # TODO: 1. What does the list of indices corresponding to "spont_black" / "spont_grey" point to?
      self.metadata = self.recording.get("db")[0][0]
      self.session_name = str(self.metadata[globals.SESSION_NAME][0])
    else:
      # Copy the attributes from another RecordingData object
      # TODO: Jupyter notebook's autoreload cannot handle this check, disable it for now
      # assert isinstance(recording_data, RecordingData)
      #self.__dict__ = deepcopy(recording_data.__dict__)
      pass

  def display_field_shapes(self):
    super().display_field_shapes()
    print("running speed", self.run_speed.shape)
    print("face motion mask", self.motion_mask.shape)
    print("face motion svd", self.motion_svd.shape)
    print("avg frame", self.avgframe.shape)
    print("pupil area", self.pupil_area.shape)
    print("pupil com", self.pupil_com.shape)
    print("eye motion svd", self.eye_motion_svd.shape)
    print("whisker motion mask", self.whisker_motion_mask.shape)
    print("whisker motion SVD", self.whisker_motion_svd.shape)

  def display_run_speed(self):
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(self.run_speed, color="orange")
    ax1.set_title("Running Speed [Session: %s]" % self.session_name, fontsize=15, fontweight='bold')
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Running Speed [mm/0.4sec]")

    ax2 = fig.add_subplot(122)
    ax2.hist(self.run_speed, bins=100, color="red", alpha=0.5)
    ax2.set_title("Running Speed Distribution", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Running Speed [mm/0.4sec]")
    ax2.set_ylabel("Occurrences")

    plt.show()

  def display_pupil_area(self):
    print("Pupil Area data NaN percentage: ", utils.nan_percentage(self.pupil_area))
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)
    ax1.plot(self.pupil_area, color="grey")
    ax1.set_title("Pupil Area [Session: %s]" % self.session_name, fontsize=15, fontweight='bold')
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Pupil Area")

    ax2 = fig.add_subplot(122)
    ax2.hist(self.pupil_area, bins=100, color="grey", alpha=0.75)
    ax2.set_title("Pupil Area Distribution", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Pupil Area")
    ax2.set_ylabel("Occurrences")

    plt.show()

  def display_aggregated_stats(self):
    fsp_mean, fsp_std = np.nanmean(self.fsp, axis=0), np.nanstd(self.fsp, axis=0)
    run_speed_mean, run_speed_std = np.nanmean(self.run_speed), np.nanstd(self.run_speed)
    pa_mean, pa_std = np.nanmean(self.pupil_area), np.nanstd(self.pupil_area)
    print("fsp:\n- mean:", fsp_mean)
    print("- std:", fsp_std)
    print("running speed:\n-mean:", run_speed_mean)
    print("- std:", run_speed_std)
    print("- range: %.2f ~ %.2f" % (np.nanmin(self.run_speed), np.nanmax(self.run_speed)))
    print("pupil area:\n- mean:", pa_mean)
    print("- std:", pa_std)
    print("- range: %.2f ~ %.2f" % (np.nanmin(self.pupil_area), np.nanmax(self.pupil_area)))

    return {"fsp": {"mean": fsp_mean, "std": fsp_std},
            "run_speed": {"mean": run_speed_mean, "std": run_speed_std},
            "pupil_area": {"mean": pa_mean, "std": pa_std}}



# TODO: Need to figure out how to align stimuli with fsp!!!
# Recording with Grating Stimuli
class GratingStimuliRecordingData(RecordingData):

  def __init__(self, recording_data=None, data_fp=None):
    super().__init__(recording_data, data_fp)
    if data_fp is not None:
      # Stimuli info
      stim = self.recording.get("stim")[0][0]
      self.stim_id = stim["istim"]  # shape: (timepoints/3, 1), integer between 0 - 33 (33 meaning grey screen)
      self.stim_direction = stim["params"]

    # stim.params: the direction of the presented drifting grating, in degrees
    # stim.paramnames: parameter that is changing across gratings (in this case the orientation of the grating)

  def display_field_shapes(self):
    super().display_field_shapes()

  def display_aggregated_stats(self):
    pass

  def auditory_tc_similarity(self):

    pass


# Recording with Image Stimuli
class VisualStimuliRecordingData(RecordingData):

  def __init__(self, recording_data=None, data_fp=None):
    super().__init__(recording_data, data_fp)
    if data_fp is not None:
      # stimtpt: binary indicator of times whether a visual stimulus was on
      self.stim_tpts = self.recording.get("stimtpt")  # shape: (Stimuli timepoints, 1

      stim = self.recording.get("stim")[0][0]
      # stimulus identity (33 means blank stimulus – gray screen)
      self.stim_id = stim["istim"]
      # average stimulus response of each neuron (stim presentations x neurons), these are averaged over 3 time bins, and are aligned to the onset of the plane in which each neuron lives
      self.stim_resp = stim["resp"]
      # times of stimulus presentations (each cell contains the stim times relative to the imaging plane, a neuron's plane is in its stat.iplane)
      self.stim_times = stim["stimtimes"]


  def display_field_shapes(self):
    super().display_field_shapes()

  def display_aggregated_stats(self):
    # Distribution of the natural images

    pass

  def visual_tc_similarity(self, sim_metric=None):
    # For each neuron, fit the stimuli to the responses


    # Compute pairwise similarity of the fitted curve


    return



