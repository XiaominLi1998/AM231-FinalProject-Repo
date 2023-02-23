import math
import numpy as np
from typing import Tuple
from copy import deepcopy

from data_loader import RecordingData, SpontaneousRecordingData
from globals import *
import plot_utils
import utils


# Note that we may need to make sure all types of the discrete predictors in the training set are observed for each sim block.
class MissingTraceSampler(object):

  def __init__(self):
    pass

  @classmethod
  def missing_fsp_consecutive_layers(self, dataset: RecordingData, sim_timestamps, sim_layer_size=1, layer_stride=1, display=False):
    """
    Simultaneous neural signals are only recorded for 'time_period' within every consecutive 'sim_layer_size' layers.
    Note: Typically, the imaging laser scans the target region with increasing depth,

    :param dataset: A RecordingData object
    :param sim_timestamps: Number of timepoints (neural signals) per simultaneous block
    :param sim_layer_size: Number of consecutive layers with simultaneous observations
    :param layer_stride: Layer increment between every two consecutive simultaneous block

    :return: fsp with missing entries set to NaN
    """
    assert sim_layer_size < dataset.layer_counts, "Please use a smaller number for sim_layer_size!"
    assert sim_timestamps < dataset.timestamp_counts, "Please use a smaller time period!"

    missing_fsp = np.full(dataset.fsp.shape, np.nan)  # fill missing entries with NaN
    sim_layers = [list(dataset.layers_zcoord[i:i+sim_layer_size]) for i in range(0, dataset.layer_counts, layer_stride)]
    total_blks, remainder = divmod(dataset.timestamp_counts, sim_timestamps)
    blk = 0
    while blk < total_blks:
      layers = sim_layers[blk % len(sim_layers)]
      start = blk * sim_timestamps
      end = start + sim_timestamps
      for lyr in layers:
        neuron_idxs = dataset.layer2neurons[lyr]
        missing_fsp[neuron_idxs, start:end] = dataset.fsp[neuron_idxs, start:end]
      blk += 1

    if remainder > 0:
      layers = sim_layers[blk % len(sim_layers)]
      for lyr in layers:
        neuron_idxs = dataset.layer2neurons[lyr]
        missing_fsp[neuron_idxs, blk*sim_timestamps:] = dataset.fsp[neuron_idxs, blk*sim_timestamps:]

    if display:
      plot_utils.display_missing_fsp(missing_fsp)
    return missing_fsp


  @classmethod
  def missing_fsp_interleaving_layers(self, dataset: RecordingData, sim_timestamps, sim_layer_size=1, display=False):
    """
    Simultaneous neural signals are only recorded for 'time_period' within 'sim_layer_size' interleaving layers.
    TODO: ask neuroscientists for confirmation if this is possible

    :param dataset: A RecordingData object
    :param sim_timestamps: Number of timepoints (neural signals) per simultaneous block
    :param sim_layer_size: Number of interleaving layers with simultaneous observations

    Note: we assume 'layer gap' splits the total number of layers as evenly as possible.
    See examples below:
        sim_layer_size=3 and layer_counts = 10, simultaneous layers: [0,3,6], [1,4,7], [2,5,8], [3,6,9]
        sim_layer_size=3 and layer_counts = 9, simultaneous layers: [0,3,6], [1,4,7], [2,5,8]
        sim_layer_size=3 and layer_counts = 8, simultaneous layers: [0,3,6], [1,4,7], [2,5]
        sim_layer_size=3 and layer_counts = 7, simultaneous layers: [0,2,4], [1,3,5], [2,4,6]

        sim_layer_size=4, layers=9, [0,2,4,6], [1,3,5,7], [2,4,6,8]
        sim_layer_size=4, layers=10, [0,3,6,9], [1,4,7], [2,5,8]    (or [0,2,4,6], [1,3,5,7], [2,4,6,8], [3,5,7,9]??)
        sim_layer_size=4, layers=11, [0,3,6,9], [1,4,7,10], [2,5,8]

    :return: fsp with missing entries set to NaN
    """
    assert sim_layer_size < dataset.layer_counts, "Please use a smaller number for sim_layer_size!"
    assert sim_timestamps < dataset.timestamp_counts, "Please use a smaller time period!"

    layer_blocks = math.ceil(dataset.layer_counts / sim_layer_size)
    gap = round(dataset.layer_counts / sim_layer_size)
    span = gap * (sim_layer_size - 1)
    sim_layers = []  # zcoord of each set of simultaneous layers
    for b in range(layer_blocks):
      layer_idxs = np.arange(b, min(b+span+1, dataset.layer_counts), gap)
      sim_layers.append(dataset.layers_zcoord[layer_idxs])

    missing_fsp = np.full(dataset.fsp.shape, np.nan)  # fill missing entries with NaN
    total_blks, remainder = divmod(dataset.timestamp_counts, sim_timestamps)
    blk = 0
    while blk < total_blks:
      layers = sim_layers[blk % layer_blocks]
      start = blk * sim_timestamps
      end = start + sim_timestamps
      for lyr in layers:
        neuron_idxs = dataset.layer2neurons[lyr]
        missing_fsp[neuron_idxs, start:end] = dataset.fsp[neuron_idxs, start:end]
      blk += 1

    if remainder > 0:
      layers = sim_layers[blk % layer_blocks]
      for lyr in layers:
        neuron_idxs = dataset.layer2neurons[lyr]
        missing_fsp[neuron_idxs, blk * sim_timestamps:] = dataset.fsp[neuron_idxs, blk * sim_timestamps:]

    if display:
      plot_utils.display_missing_fsp(missing_fsp)
    return missing_fsp


  @classmethod
  def sample_missing_traces_by_surface_region(self, neuron_traces, region_xrange: Tuple, region_yrange: Tuple, overlap_pct=0.5, ratio=True):
    # Simultaneous neuron blocks are defined by neuron distance of the same layer

    pass


  @classmethod
  def sample_missing_traces_by_blocks(self, dataset: RecordingData, neuronblock_size, timeblock_size, overlap_percent=0.5, display=False):
    """
    The simultaneous blocks are grouped by the blocks of neuron indices of the input dataset.

    Note: If user needs to specify a time range of interest, the input dataset should be chunked.

    :param dataset:
    :param timeblock_size: Number of timepoints (neural signals) per simultaneous block
    :param neuronblock_size: number of neurons to per simulated calcium imaging snapshot
    :param overlap_pct: Percentage of overlapping neurons in each block
    :return:
    """
    # if not isinstance(dataset, (RecordingData, PartialRecordingDataset)):
    #   print("Input dataset has to be a RecordingData or its subclass!")
    #   return

    # Load params from the dataset
    neuron_counts = dataset.neuron_counts
    ts_counts = dataset.timestamp_counts
    full_fsp = dataset.fsp
    print("subset fsp: ", full_fsp.shape)

    # Get indices of neuron blocks
    block_stepsize = int(neuronblock_size * (1 - overlap_percent))
    neuron_start = 0
    neighborhoods = []
    while neuron_start <= neuron_counts - neuronblock_size:
      neighborhoods.append(neuron_start + np.arange(neuronblock_size))
      neuron_start += block_stepsize
    if 0 < neuron_start < neuron_counts:  # TODO: Signal the truncation!
      print("The last %d neurons (less than a neuron block) are grouped together!" %
            (neuron_counts - neuron_start))
      neighborhoods.append(np.arange(neuron_start, neuron_counts))

    # Save observed blocks across time period
    print("Number of Simultaneous Neighborhoods: ", len(neighborhoods))
    missing_fsp = np.zeros_like(full_fsp) + np.nan  # Missing entries are NA
    nbrhood_idx = 0
    for start in range(0, ts_counts, timeblock_size):
      end = min(start + timeblock_size, dataset.timestamp_counts)
      missing_fsp[neighborhoods[nbrhood_idx], start:end] = full_fsp[neighborhoods[nbrhood_idx], start:end]
      nbrhood_idx = (nbrhood_idx + 1) % len(neighborhoods)

    if display:
      plot_utils.display_missing_fsp(missing_fsp)
    return missing_fsp



class PartialSpontRecordingData(SpontaneousRecordingData):

  def __init__(self, dataset: SpontaneousRecordingData, *, layers='all',
               start_timepoint=None, end_timepoint=None,
               x_range: Tuple[float, float], y_range: Tuple[float, float], ratio=True):
    """
    This class samples a partial recordings from an original calcium imaging dataset.

    :param dataset: File path to the dataset to load the original RecordingData
    :param layers: A list of layers to include in this PartialRecordingDataset
    :param start_timepoint:
    :param end_timepoint:
    :param ul_coord:
    :param lr_coord:
    """
    super().__init__(recording_data=dataset)  # void statement that does nothing
    self.original_dataset = dataset
    # if type(dataset) is RecordingData or isinstance(dataset, dl.SpontaneousRecordingData):
    #   self.original_dataset = dataset
    # else:
    #   raise TypeError("Original dataset must be a RecordingData object!") TODO: jupyter notebook keeps failing here

    # Check args - temporal dimension
    start_ts, end_ts = 0, dataset.timestamp_counts
    if start_timepoint is not None and end_timepoint is not None:
      if not (0 <= start_timepoint < dataset.timestamp_counts):
        raise ValueError("Start timepoint must lie in [0, %d)!" % dataset.timestamp_counts)
      if not (start_timepoint < end_timepoint <= dataset.timestamp_counts):
        raise ValueError("End timepoint must lie in (start_tp=%d, %d]!" % (start_timepoint, dataset.timestamp_counts))
      start_ts = max(0, start_timepoint)
      end_ts = min(dataset.timestamp_counts, end_timepoint)

    # Check args - spatial dimensions
    if ratio:
      if not (0 <= x_range[0] < 1): raise ValueError("Min(x) in ratio should lie in (0, 1.0]!")
      if not (0 < x_range[1] <= 1): raise ValueError("Max(x) in ratio should lie in (0, 1.0]!")
      if not (0 <= y_range[0] < 1): raise ValueError("Min(y) in ratio should lie in (0, 1.0]!")
      if not (0 < y_range[1] <= 1): raise ValueError("Max(y) in ratio should lie in (0, 1.0]!")

    if layers != 'all' and (max(layers) >= dataset.layer_counts or min(layers) < 0):
      raise ValueError("Layers must be between 0 and %d!" % dataset.layer_counts)
      # TODO: check if neuron_xyz has z in ascending order by default (add helper)

    # Vertical axis (layer)
    if layers == 'all':
      self.layers_zcoord = deepcopy(dataset.layers_zcoord)
      self.layer_counts = len(self.layers_zcoord)
      #self.layer2neurons = deepcopy(dataset.layer2neurons)
      #self.neuron_xyz = deepcopy(dataset.neuron_xyz)
    else:
      layers = sorted(list(set(layers)))  # make sure there's not duplicates and 'layers' is sorted ascendingly
      self.layers_zcoord = dataset.layers_zcoord[layers]
      self.layer_counts = len(self.layers_zcoord)

    # Horizontal spatial region (x, y)
    new_fsp, new_xyz = [], []
    new_layer2neurons = {}

    if ratio:
      # Apply the spatial boundary to each layer of neurons
      for z, neurons_idx in dataset.layer2neurons.items():
        if z not in self.layers_zcoord:
          continue
        neuronXY_z = dataset.neuron_xyz[neurons_idx, :]
        minX, minY = np.min(neuronXY_z, axis=0)[:2]
        maxX, maxY = np.max(neuronXY_z, axis=0)[:2]
        lx = minX + x_range[0] * (maxX - minX)
        rx = minX + x_range[1] * (maxX - minX)
        ly = minY + y_range[0] * (maxY - minY)
        ry = minY + y_range[1] * (maxY - minY)
        # Select the neurons within the rectangular region
        selected_neurons_idx = np.where((lx <= neuronXY_z[:, 0]) & (neuronXY_z[:, 0] <= rx) &
                                        (ly <= neuronXY_z[:, 1]) & (neuronXY_z[:, 1] <= ry))[0]  # wrt segmented neuronXY_z
        new_layer2neurons[z] = np.arange(len(new_xyz), len(new_xyz) + len(selected_neurons_idx), step=1)
        new_xyz.extend(neuronXY_z[selected_neurons_idx])
        new_fsp.extend(dataset.fsp[neurons_idx[selected_neurons_idx], :])
    else:
      for z, neurons_idx in dataset.layer2neurons.items():
        if z not in self.layers_zcoord:
          continue
        neuronXY_z = dataset.neuron_xyz[neurons_idx, :]
        minX, minY = np.min(neuronXY_z, axis=0)[:2]
        maxX, maxY = np.max(neuronXY_z, axis=0)[:2]
        lx = max(minX, x_range[0])
        rx = min(maxX, x_range[1])
        ly = max(minY, y_range[0])
        ry = min(maxY, y_range[1])
        # Select the neurons within the rectangular region
        selected_neurons_idx = np.where((lx <= neuronXY_z[:, 0]) & (neuronXY_z[:, 0] <= rx) &
                                        (ly <= neuronXY_z[:, 1]) & (neuronXY_z[:, 1] <= ry))[0]
        new_layer2neurons[z] = np.arange(len(new_xyz), len(new_xyz) + len(selected_neurons_idx), step=1)
        new_xyz.extend(neuronXY_z[selected_neurons_idx])
        new_fsp.extend(dataset.fsp[neurons_idx[selected_neurons_idx], :])

    self.fsp = np.array(new_fsp)
    self.neuron_counts = self.fsp.shape[0]
    self.neuron_xyz = np.array(new_xyz)
    self.layer2neurons = new_layer2neurons


    # Temporal dimension (arrays are not deepcopied because they should be read-only)
    self.fsp = self.fsp[:, start_ts:end_ts]
    self.timestamp_counts = end_ts - start_ts
    self.timestamp_offset = start_timepoint
    self.run_speed = dataset.run_speed[start_ts:end_ts, :]
    self.motion_svd = dataset.motion_svd[start_ts:end_ts, :]
    self.whisker_motion_svd = dataset.whisker_motion_svd[start_ts:end_ts, :]
    self.eye_motion_svd = dataset.eye_motion_svd[start_ts:end_ts, :]
    self.pupil_area = dataset.pupil_area[start_ts:end_ts, :]
    self.pupil_com = dataset.pupil_com[start_ts:end_ts, :]

    self.motion_mask = dataset.motion_mask
    self.whisker_motion_mask = dataset.whisker_motion_mask
    self.avgframe = dataset.avgframe
    self.session_name = "Partial " + dataset.session_name

  # TODO: Might need partial trace list, grouped by simultanous time period

  def get_partial_fsp(self, starttime=0, endtime=np.inf, display=False):
    return np.array(self.fsp)

  def get_original_trace(self):
    return np.array(self.original_dataset.fsp)

  def get_original_time_period(self):
    return (0, self.original_dataset.timestamp_counts)

  def get_original_neuron_xyz(self):
    return np.array(self.original_dataset.neuron_xyz)
