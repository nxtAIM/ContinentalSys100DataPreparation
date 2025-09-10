import sys, os
import h5py
import numpy as np
from threading import Thread, Event
from collections import deque
from queue import Queue, Empty
import multiprocessing as mp
import psutil
import time
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import socket
import traceback
from difflib import ndiff

import pyarrow.parquet as pq

sys.path.append(os.path.dirname(__file__))
import radarDiscretizer as RD
sys.path.append(os.path.join(os.path.dirname(__file__), 'sys100/src/proto'))

from google.protobuf.descriptor import FieldDescriptor
import GPSDevice_pb2
import VehDyn_pb2
import SPoseCalibration_pb2
import SPoseDynamic_pb2
import SensorMounting_pb2
import VehPar_pb2
import RSP2_ClusterListNS_pb2
import ClustListHead_pb2


def get_available_cpus():
    # better method to get the number of available CPUs than multiprocessing.cpu_count()
    return len(os.sched_getaffinity(0))

def format_seconds(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def string_diff(s1,s2):
    diff = ndiff(s1,s2)
    plus = ""
    minus = ""
    for line in diff:
        if line.startswith('+'):
            plus += line[2:]  # No need to .strip() — this removes actual chars
        elif line.startswith('-'):
            minus += line[2:]
    return minus, plus

def RSPClustersToMeasurments(measurement):
    """
    returns range, azimuth0, azimuth1, VrelRad and RCS from a measurement
    all are (N,)
    """
    if measurement is None:
        return None
    clusters = measurement.a_RSPClusters

    # Preallocate arrays for performance
    num_clusters = len(clusters)
    r = np.empty(num_clusters, dtype=np.float32)
    azimuth0 = np.empty(num_clusters, dtype=np.float32)
    azimuth1 = np.empty(num_clusters, dtype=np.float32)
    VrelRad = np.empty(num_clusters, dtype=np.float32)
    RCS = np.empty(num_clusters, dtype=np.float32)

    for i, cluster in enumerate(clusters):
        azimuth0[i] = cluster.a_AzAng[0]
        azimuth1[i] = cluster.a_AzAng[1]
        r[i]        = cluster.f_RangeRad
        VrelRad[i]  = cluster.f_VrelRad
        RCS[i]      = cluster.f_RcsRaw

    return r, azimuth0, azimuth1, VrelRad, RCS

def RSPClustersToPoints(measurement):
    """
    returns points, azimuth1, VrelRad and RCS from a measurement
    points are (3 x N), azimuth1, VrelRad and RCS are (N,)
    x points forward, y points left, z points up
    """
    if measurement is None:
        return None
    r, azimuth0, azimuth1, VrelRad, RCS = RSPClustersToMeasurments(measurement)
    
    x = np.cos(azimuth0) * r
    y = np.sin(azimuth0) * r
    z = 0 * y
    points = np.vstack([x, y, z])
    return points, azimuth0, azimuth1, VrelRad, RCS

def deserialialize_protobuf(bytestring, message_class):
    """ Takes a bytestring instance representation of a message object class
    and derializes into a python object"""
    message = message_class()
    if bytestring is not None:
        message.ParseFromString(bytestring)
    return message

proto2np_dtypes = {
    FieldDescriptor.TYPE_DOUBLE: np.float64,
    FieldDescriptor.TYPE_FLOAT:  np.float32,
    FieldDescriptor.TYPE_INT64:  np.int64,
    FieldDescriptor.TYPE_UINT64: np.uint64,
    FieldDescriptor.TYPE_INT32:  np.int32,
    FieldDescriptor.TYPE_UINT32: np.uint32,
    FieldDescriptor.TYPE_BOOL:   np.uint8,
}
default_values = {
    FieldDescriptor.TYPE_DOUBLE: np.float64('nan'),
    FieldDescriptor.TYPE_FLOAT:  np.float32('nan'),
    FieldDescriptor.TYPE_INT64:  np.iinfo(np.int64).min,
    FieldDescriptor.TYPE_UINT64: np.iinfo(np.uint64).max,
    FieldDescriptor.TYPE_INT32:  np.iinfo(np.int32).min,
    FieldDescriptor.TYPE_UINT32: np.iinfo(np.uint32).max,
    FieldDescriptor.TYPE_BOOL:   2,
}

def protobuf_to_dtype(message_descriptor):
    fields = []
    for field in message_descriptor.fields:
        name = field.name
        if field.label == FieldDescriptor.LABEL_REPEATED:
            raise ValueError(f"Repeated fields with unkown length are not supported: {name}")
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            sub_dtype = protobuf_to_dtype(field.message_type)
            fields.append((name, sub_dtype))
        else:
            np_dtype = proto2np_dtypes.get(field.type, None)
            if np_dtype is None:
                raise NotImplementedError(f"Field type {field.type} not implemented.")
            fields.append((name, np_dtype))
    return np.dtype(fields)

def npget(*args):
    """ Get values from a numpy structured array like
            structured_array = np.array([
                ('Alice', 30, 1.6),
                ('Bob', 25, 1.8),
                ('Charlie', 35, 1.9)
            ], dtype=[('name', 'U10'), ('age', 'i4'), ('height', 'f4')])
        In this case 'npget(structured_array, 'name') would return array(['Alice', 'Bob', 'Charlie'], dtype='<U10'),
                     'npget(structured_array, 'parent') would raise a ValueError and
                     'npget(structured_array, 'parent', None) would return None
    :param 1: first param is the array
    :param 2: second param is the index or key
    :optional param 3: the default value in case the key does not exists
    """
    try:
        return args[0][args[1]]
    except ValueError as e:
        if len(args) >= 3:
            return args[2]
        raise e

def pprint_structnp(arr, pad=""):
    """ Pretty print structured and unstructured NumPy arrays """
    if arr is None:
        print(pad + "None")
        return
    if not (isinstance(arr, np.ndarray) or isinstance(arr, np.void)):
        print(pad + str(arr))
        return

    # Structured array
    if arr.dtype.names:
        if arr.shape != ():  # Not a scalar
            maxspace = sum([np.ceil(np.log10(s))+2 for s in arr.shape]) + (1 if len(arr.shape) == 1 else 0)
            for idx, item in np.ndenumerate(arr):
                pprint_structnp(item, pad=pad+f"  {idx:>{maxspace}}:  ")
        else:  # Scalar structured
            for name in arr.dtype.names:
                value = arr[name]
                if (isinstance(arr, np.ndarray) or isinstance(arr, np.void)) and value.dtype.names:
                    # Nested structured array
                    print(pad + f"{name} ({value.itemsize} bytes):")
                    pprint_structnp(value, pad=pad+"  ")
                elif isinstance(value, np.ndarray):
                    # Regular ndarray
                    display_value = str(value).replace("\n", "\\n ")
                    if len(display_value) > 100:
                        display_value = display_value[:100] + " ..."
                    shapestr = f"{value.shape}, " if value.ndim > 0 else ""
                    print(pad + f"{name} ({shapestr}{value.dtype}): {display_value}")
                else:
                    print(pad + f"{name} ({str(type(value)).replace('class ', '')}): {value}")
    else:
        print(pad + str(arr).replace("\n", "\\n "))

def protobuf_to_numpy(message, cast_directly=False, out=None):
    data = []
    dtypes = []
    for field in message.DESCRIPTOR.fields:
        if field.type == FieldDescriptor.TYPE_MESSAGE:
            if not field.has_presence or message.HasField(field.name):
                sub_message = getattr(message, field.name)
            else:
                sub_message = field.message_type._concrete_class()
            
            if field.label == FieldDescriptor.LABEL_REPEATED:
                if out is not None:
                    sub_message = [protobuf_to_numpy(msg, cast_directly=cast_directly,
                                                     out=out[field.name][i]) 
                                   for i,msg in enumerate(sub_message)]
                else:
                    if not cast_directly:
                        raise NotImplementedError(f"Repeated field {field.name} not supported without cast_directly=True.")
                    sub_message = [protobuf_to_numpy(msg, cast_directly=True) 
                                   for msg in sub_message]
                    data.append( sub_message )
                    dtypes.append( (field.name, sub_message[0].dtype, (len(sub_message),)) )
            else:
                if out is not None:
                    sub_message = protobuf_to_numpy(sub_message, cast_directly=cast_directly,
                                                    out=out[field.name])
                else:
                    sub_message = protobuf_to_numpy(sub_message, cast_directly=cast_directly)
                    data.append( sub_message )
                    if cast_directly:
                        dtypes.append( (field.name, sub_message.dtype) )
        
        elif field.label == FieldDescriptor.LABEL_REPEATED:
            if out is not None:
                out[field.name] = np.array([msg_i for msg_i in getattr(message, field.name)],
                                           dtype=proto2np_dtypes[field.type])
            elif cast_directly:
                msgs = getattr(message, field.name)
                data.append(msgs)
                dtypes.append( (field.name, proto2np_dtypes[field.type], (len(msgs),)) )
            else:
                raise NotImplementedError(f"Repeated field {field.name} not supported without cast_directly=True.")
        else:
            if not field.has_presence or message.HasField(field.name):
                value = getattr(message, field.name)
            else:
                value = default_values[field.type]
            if out is not None:
                out[field.name] = value
            else:
                if cast_directly:
                    dtypes.append( (field.name, proto2np_dtypes[field.type]) )
                data.append( value )
    if out is not None:
        return out
    if cast_directly:
        dtype = np.dtype([t for d,t in zip(data, dtypes) if not hasattr(d,'itemsize') or d.itemsize > 0])
        return np.array(tuple([d for d in data if not hasattr(d,'itemsize') or d.itemsize > 0]), 
                        dtype=dtype)
    return tuple(data)


# Writer thread for each process
class HDF5Writer(Thread):
    def __init__(self, queue, file_path, 
                 dataset_shape, chunk_shape, dtype=np.float32,
                 prefix="", log_freq=10000):
        super().__init__()
        self.queue = queue
        self.file_path = file_path
        self.dataset_shape = dataset_shape
        self.chunk_shape = chunk_shape
        self.dtype = dtype
        self.log_freq = log_freq
        #self.running = True
        self._stop_event = Event()
        self.total_frames = dataset_shape[0]
        self.prefix = prefix
        self.write_count = 0
        self.start_time = time.time()
        self.movingaverage = 0
        self.sincelastupdate = 0

    def message(self, finished=False):
        now = time.time()
        duration = now - self.start_time
        estimated_remaining = (self.total_frames - self.write_count) / self.write_count * duration
        estimated_total = (self.total_frames / self.write_count) * duration
        mem = psutil.virtual_memory()
        print(
            f"{self.prefix} {format_seconds(now)}: " +
            f"Iteration {self.write_count:>5}/{self.total_frames}: " +
            f"[{format_seconds(duration)} < {format_seconds(estimated_remaining)} | {format_seconds(estimated_total)}" +
            f", {self.write_count / duration:.2f} it/s]{' finished' if finished else ''} " +
            f"RAM {mem.percent:.1f}% {mem.used / 1e9:.2f}/{mem.total / 1e9:.2f} GB "+
            f"- this process {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB "+
            f"average Queue size {self.movingaverage/self.sincelastupdate:.1f}"+
            f"/{self.queue.maxsize}",
            flush=True
        )
        self.movingaverage = 0
        self.sincelastupdate = 0


    def run(self):
        self.start_time = time.time()
        with h5py.File(self.file_path, 'w') as f:
            dataset = f.create_dataset(
                'recordings',
                shape=self.dataset_shape,
                chunks=self.chunk_shape,
                dtype=self.dtype,
                compression='lzf',
            )
            while (not self._stop_event.is_set()) or not self.queue.empty():
                try:
                    didx, idx, data = self.queue.get()
                    self.movingaverage += self.queue.qsize()
                    self.sincelastupdate += 1
                    if didx == 0:
                        dataset[idx, ...] = data
                        self.write_count += 1
                        if self.write_count % self.log_freq == 0:
                            self.message(finished=False)
                        self.queue.task_done()
                    elif didx == 2:
                        # set metadata
                        for name, value in data:
                            #print(f"try to set metadata '{name}' with value")
                            #pprint_structnp(value, pad="  ")
                            f.attrs[name] = value
                        self.queue.task_done()

                except Empty:
                    time.sleep(0.1) # Avoid busy waiting even though we also wait in the queue
            self.message(finished=True)
    
    def stop(self):
        self._stop_event.set()
        #self.running = False


def is_valid_h5_file(filepath, expected_shape, expected_dtype, prefix=""):
    if not os.path.exists(filepath):
        return False
    try:
        with h5py.File(filepath, 'r') as f:
            if 'valid_entries' not in f.attrs:
                print(f"{prefix} file exists but missing 'valid_entries' attribute. Reprocessing.", flush=True)
                return False
            if 'recordings' not in f:
                print(f"{prefix} file exists but missing 'recordings' dataset. Reprocessing.", flush=True)
                return False
            dataset = f['recordings']
            if dataset.shape != expected_shape:
                s1, s2 = str(dataset.shape), str(expected_shape)
                if len(s1)+len(s2) < 200:
                    out = f"{s1}, expected {s2}"
                else:
                    old,new = string_diff(s1,s2)
                    out = f" ... '{old}' -> '{new}'"
                print(f"{prefix} preexisting file has incorrect discretised radar shape {out}. Reprocessing.", flush=True)
                return False
            if dataset.dtype != expected_dtype:
                s1, s2 = str(dataset.dtype), str(expected_dtype)
                if len(s1)+len(s2) < 200:
                    out = f"{s1}, expected {s2}"
                else:
                    old,new = string_diff(s1,s2)
                    out = f" ... '{old}' -> '{new}'"
                print(f"{prefix} preexisting file has incorrect measurements dtype '{out}'. Reprocessing.", flush=True)
                return False
            
    except Exception as e:
        print(f"{prefix} Failed to open or read HDF5 file: {e}. Reprocessing.")
        return False
    return True


def write_radarh5_from_parquet(parquet_path,
                               rank=None,
                               path_index=0,
                               total_paths=1,
                               use_tqdm=True,
                               log_freq=10000,
                               chunk_size:int=16,
                               xmin:int=0, xmax:int=102, 
                               ymin:int=-100, ymax:int=100,
                               discretization_steps:int=64,
                               valid_indicator=None,
                               max_file_size=None):
    prefix = (f"[{rank}] " if rank is not None else "") + f"{path_index:>{len(str(total_paths))}}/{total_paths} (PID {os.getpid():>8}):"
    filename = os.path.basename(parquet_path).replace('.parquet','')
    parent_dir = os.path.dirname(parquet_path)
    mem = psutil.virtual_memory()

    ds = discretization_steps
    try:
        outpath = parquet_path.replace('.parquet', f'_radargrids_{ds}.h5')
        
        parquet_file = pq.ParquetFile(parquet_path)
        frames = parquet_file.metadata.num_rows
        if max_file_size is not None:
            frames = min(frames, max_file_size)

        expected_mes_dtype = np.dtype([
            ('ImageTimestamp',    np.int64),
            ('ImageFrameCounter', np.int64),
            ('VideoId',           np.int32),
            ('VideoFrameCounter', np.int64),
            ('RadarFrame',        np.float32, (ds, ds, 7)),
            ('RadarClusterHead',  protobuf_to_dtype(ClustListHead_pb2.ClustListHead.DESCRIPTOR)),
            ('CameraPoseCalibration', protobuf_to_dtype(SPoseCalibration_pb2.SPoseCalibration.DESCRIPTOR)),
            ('CameraPoseDynamic', protobuf_to_dtype(SPoseDynamic_pb2.SPoseDynamic.DESCRIPTOR)),
            ('GPS', protobuf_to_dtype(GPSDevice_pb2.GPSDevice.DESCRIPTOR)),
            ('VehDyn', protobuf_to_dtype(VehDyn_pb2.VehDyn.DESCRIPTOR)),
            ('VehDyn_Camera', protobuf_to_dtype(VehDyn_pb2.VehDyn.DESCRIPTOR)),
        ])
        video_suffixes = {}

        if is_valid_h5_file(outpath, (frames,), expected_mes_dtype, prefix=prefix):
            print(f"{prefix} {format_seconds(time.time())} Skipping '{filename}', " +
                  f"HDF5 of {frames} frames already valid, RAM {mem.percent:.1f}% " +
                  f"{mem.used / 1e9:.2f}/{mem.total / 1e9:.2f} GB - this process " +
                  f"{psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB",
                  flush=True)
            return 2

        print(f"{prefix} {format_seconds(time.time())} found {frames} samples in '{filename}', " +
              f"parquet with file size: {os.path.getsize(parquet_path) / (1024 * 1024 * 1024)} GB, " +
              f"RAM {mem.percent:.1f}% {mem.used / 1e9:.2f}/{mem.total / 1e9:.2f} GB - " +
              f"this process {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB",
              flush=True)
        
        discretizer = RD.RadarDiscretizer(
            xmin=xmin, xmax=xmax, 
            ymin=ymin, ymax=ymax, 
            discretization_steps=ds,
            valid_indicator=valid_indicator,
        )
        
        queue = Queue(maxsize=max(4*chunk_size,512))  # Small buffer for I/O burst
        writer = HDF5Writer(
            queue=queue,
            file_path=outpath,
            dataset_shape=(frames,),
            chunk_shape=(chunk_size,),
            dtype=expected_mes_dtype,
            prefix=prefix,
            log_freq=log_freq,
        )
        writer.start()

        progress = tqdm(total=frames) if use_tqdm else None

        cumidx = 0
        groups = -1
        for batch in parquet_file.iter_batches():
            batch_df = batch.to_pandas()
            groups += 1
            for idx, row in batch_df.iterrows():
                if max_file_size is not None and cumidx >= max_file_size:
                    print(f"{prefix} reached max file size of {max_file_size} frames after row_idx {idx-1}, stopping processing", flush=True)
                    break
                if row.ARS5xx_RawDataCycle_RSP2_ClusterListNS is None:
                    print(f"{prefix} WARNING: at {filename} no RSPClusters found for timestamp {row.Image_Timestamp}/rowgroup {groups} idx {idx}", flush=True)
                    continue
                RSPClusters = deserialialize_protobuf(row.ARS5xx_RawDataCycle_RSP2_ClusterListNS, 
                                                    RSP2_ClusterListNS_pb2.RSP2_ClusterListNS)
                RSPClusterHead = RSPClusters.clustListHead
                
                if not RSPClusterHead.HasField('f_RangeLimit'):
                    if not RSPClusterHead.HasField('f_RangeGateLength'):
                        print(f"{prefix} WARNING: at {filename} no f_RangeGateLength found for {row.Image_Timestamp}/rowgroup {groups} idx {idx}", flush=True)
                        continue
                    rangeGateLength = RSPClusterHead.f_RangeGateLength
                else:
                    rangeGateLength = RSPClusterHead.f_RangeLimit / 256
                
                if not RSPClusterHead.HasField('f_AmbFreeDopplerRange'):
                    dopplerRange = 20
                else:
                    dopplerRange = RSPClusterHead.f_AmbFreeDopplerRange
                
                
                points, azimuth0, azimuth1, vel, RCS = RSPClustersToPoints(RSPClusters)

                # put the rangeGateLength into the data and adjust the discretizer
                
                discretizer.xmax = rangeGateLength * 256
                points[-1, :] = rangeGateLength
                
                """
                the radar grid will be:
                 - x_offset, 
                 - y_offset,
                 - rangeGateLength, 
                 - azimuth1, 
                 - RadarCrossSection, 
                 - Relative radial velocity,
                 - Expected static relative radial velocity
                """
                radar_grid = np.zeros((ds, ds, 6))
                #centre_x = (np.arange(ds) + .5)*discretizer.xstep + discretizer.xmin
                #centre_y = (np.arange(ds) + .5)*discretizer.ystep + discretizer.ymin
                #radar_grid[...,6] = np.arctan2(centre_y, centre_x)

                radar_frame = discretizer.to_grid(
                    points=np.vstack([points, azimuth1, RCS, vel]).T,
                    grid=radar_grid
                ).astype(np.float32)
                #assert radar_frame.shape == (ds, ds, 7), f"Unexpected radar frame shape: {radar_frame.shape}"

                vehDyn = deserialialize_protobuf(row.ARS5xx_AlgoVehCycle_VehDyn, VehDyn_pb2.VehDyn)
                velocity = vehDyn.longitudinal.Velocity

                # calculate the expected static relative radial velocity
                disc_x = (radar_frame[...,0] + np.arange(ds)[:,None] + 0.5)
                disc_x = discretizer.xmin + disc_x * discretizer.xstep 
                disc_y = (radar_frame[...,1] + np.arange(ds)[None,:] + 0.5)
                disc_y = discretizer.ymin + disc_y * discretizer.ystep
                
                staticVrelRad = -velocity * disc_x / np.sqrt(disc_x**2 + disc_y**2)
                staticVrelRad -= np.floor(1/2+staticVrelRad/dopplerRange)*dopplerRange

                video_suffix_id = video_suffixes.get(row.Video_Suffix, None)
                if video_suffix_id is None:
                    video_suffix_id = len(video_suffixes)
                    video_suffixes[row.Video_Suffix] = video_suffix_id
                
                measurement = np.array( (
                    row.Image_Timestamp,
                    row.Image_FrameCounter,
                    video_suffix_id,
                    row.Video_FrameCounter -1, # originally 1-based
                    np.concatenate([radar_frame, staticVrelRad[:,:,None]],axis=2), # the discretised radar frame
                    protobuf_to_numpy(RSPClusterHead,
                        out=np.zeros((), dtype=expected_mes_dtype['RadarClusterHead'])
                    ),
                    protobuf_to_numpy(deserialialize_protobuf(
                            row.MFC5xx_Device_ACAL_pMonoCalibration_SPoseCalibration,
                            SPoseCalibration_pb2.SPoseCalibration),
                        out=np.zeros((), dtype=expected_mes_dtype['CameraPoseCalibration'])
                    ),
                    protobuf_to_numpy(deserialialize_protobuf(
                            row.MFC5xxDevice_ACAL_pMonoCalibration_SPoseDynamic,
                            SPoseDynamic_pb2.SPoseDynamic),
                        out=np.zeros((), dtype=expected_mes_dtype['CameraPoseDynamic'])
                    ),
                    protobuf_to_numpy(deserialialize_protobuf(
                            row.GPSDevice, GPSDevice_pb2.GPSDevice),
                        out=np.zeros((), dtype=expected_mes_dtype['GPS'])
                    ),
                    protobuf_to_numpy(vehDyn,
                        out=np.zeros((), dtype=expected_mes_dtype['VehDyn'])
                    ),
                    protobuf_to_numpy(deserialialize_protobuf(
                            row.MFC5xx_Device_VDY_VehDyn, VehDyn_pb2.VehDyn),
                        out=np.zeros((), dtype=expected_mes_dtype['VehDyn_Camera'])
                    ),
                ), dtype=expected_mes_dtype)

                queue.put((0, cumidx, measurement))
                if use_tqdm:
                    progress.update(1)
            
                cumidx += 1
        
        # Write remaining metadata
        CameraMounting = protobuf_to_numpy(deserialialize_protobuf(
            # if we don't have valid values at the end of the recording, there are none
            row.MFC5xx_Device_NVMAD_sRamDataFreeze_RAM_NVM_BLOCK_IPC_e_CL_IUC_VEHICLE_PARAMETERS_SensorMounting,
            SensorMounting_pb2.SensorMounting
        ), cast_directly=True)
        VehPar = protobuf_to_numpy(deserialialize_protobuf(
            # if we don't have valid values at the end of the recording, there are none
            row.ARS5xx_AlgoVehCycle_VehPar,
            VehPar_pb2.VehPar
        ), cast_directly=True)

        #write metadata
        video_paths = {id: f"{filename}{suff}" for suff, id in video_suffixes.items()}
        queue.put((2, -1, [('valid_entries', cumidx),
                           ('VideoId2Path', [video_paths[id] for id in range(len(video_suffixes))]),
                           ('CameraMounting', CameraMounting),
                           ('VehPar', VehPar),
                          ]))

        # Tell writer to stop and flush
        if use_tqdm:
            progress.close()
        writer.stop()
        queue.join()
        writer.join()
        print(f"{prefix} Finished writing '{filename}'", flush=True)
        return 1
    except Exception as e:
        print(f"{prefix} ERROR processing '{filename}': {e}", flush=True)
        raise e


# Entry point
# For running on login nodes, limited to 20 CPUs: (only 20 processes are allowed on a login node, nohup removes the dependency on your SSH connection to stay stable)
# nohup nice -n 19 taskset -c 0-19 python radar_src/discretise_radar_as_h5.py > radar_preprocessing_output.txt 2>&1 &
# cancel all processes again with "pkill -f discretise_radar_as_h5.py" or "pkill -u {your_username}" (pkill -u neuhoefer1)
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parquet_paths = [
        "/p/data1/nxtaim/proprietary/continental/sys100/2021.04.17_at_05.10.53/2021.04.17_at_05.10.53_camera-radar-mi_5374.parquet",
        "/p/data1/nxtaim/proprietary/continental/sys100/2021.04.15_at_11.18.03/2021.04.15_at_11.18.03_camera-radar-mi_5316.parquet",
        "/p/data1/nxtaim/proprietary/continental/sys100/2021.07.07_at_23.42.23/2021.07.07_at_23.42.23_camera-radar-mi_5374.parquet",
    ]
    usetqdm = False
    reverse = False
    logfreq = 1000
    rank = None
    node = socket.gethostname()
    print(f"seeing {mp.cpu_count()} CPUs / {get_available_cpus()} available CPUs on node: {node}")
    num_workers = min(len(parquet_paths), get_available_cpus())
    if True:
        # THIS NEEDS TO BE SET TO YOUR PATH
        path_gen = Path("/p/data1/nxtaim/proprietary/continental/sys100").rglob("*.parquet")
        
        if len(sys.argv) == 3:
            rank = int(sys.argv[1])
            world_size = int(sys.argv[2])
        else:
            world_size = os.environ.get("SLURM_NTASKS", None)
            rank = os.environ.get("SLURM_PROCID", None)
        if world_size is not None and rank is not None:
            world_size = int(world_size)
            rank = int(rank)
            parquet_paths = [str(path) for idx, path in enumerate(path_gen) if idx % world_size == rank]
            num_workers = int(os.getenv('SRUN_CPUS_PER_TASK', min(len(parquet_paths), get_available_cpus())))
            print(f"{format_seconds(time.time())}: Process {rank}/{world_size} processes {len(parquet_paths)} files with {num_workers} workers ({mp.cpu_count()} total, {get_available_cpus()} made available)")
            rank = f"{rank:>{len(str(world_size))}}"
        else:
            parquet_paths = [str(path) for path in path_gen]
            num_workers = int(os.getenv('SRUN_CPUS_PER_TASK', min(len(parquet_paths), get_available_cpus())))
            print(f"{format_seconds(time.time())}: processing {len(parquet_paths)} files with {num_workers} workers ({mp.cpu_count()} total, {get_available_cpus()} made available)")
    else:
        print(f"{format_seconds(time.time())}: processing {len(parquet_paths)} files with {num_workers} workers ({mp.cpu_count()} total, {get_available_cpus()} made available)")
    
    # Shared progress counter
    TOTAL_TASKS = len(parquet_paths)
    counter = mp.Value('i', 0)
    counter_lock = mp.Lock()
    def callback_done(result):
        with counter_lock:
            counter.value += 1
            print(f"[{format_seconds(time.time())}] Global Progress {counter.value}/{TOTAL_TASKS}", flush=True)
    def error_callback(e):
        print(f"[ERROR] A worker failed with: {e}", flush=True)
        traceback.print_exception(type(e), e, e.__traceback__, file=sys.stdout)
    
    with mp.Pool(processes=max(num_workers, 1)) as pool:
        results = []
        iterator = enumerate(parquet_paths) if reverse==False else reversed(list(enumerate(parquet_paths)))
        for idx, path in iterator:
            res = pool.apply_async(
                write_radarh5_from_parquet,
                args=(path, rank, idx, len(parquet_paths), usetqdm, logfreq),
                callback=callback_done,
                error_callback=error_callback
            )
            results.append(res)
        
        for idx, r in enumerate(results):
            r.wait()
            print(f"The {'first' if reverse==False else 'last'} {idx} jobs are done", flush=True)

        pool.close()
        pool.join()
    print("Pool context exited.")