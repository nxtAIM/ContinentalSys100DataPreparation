import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pandas as pd
import pyarrow.parquet as pq
from google.protobuf.json_format import MessageToDict
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'sys100/src/proto'))

from time import time

import GPSDevice_pb2
import VehDyn_pb2
import SPoseCalibration_pb2
import SPoseDynamic_pb2
import SSigHeader_pb2
import SensorMounting_pb2
import VehPar_pb2
import RSP2_ClusterListNS_pb2

def deserialialize_protobuf(series, message_class):
    """ Takes a series from a parquet table and a message object class
    and derializes into a list of python objects"""
    result = []
    for value in series.values:
        message = message_class()
        if value is not None:
            message.ParseFromString(value)
        result.append(message)

    return result

def to_dataframe(messages):
    """Takes a list of deserialized protobuf messages,
    i.e. python objects and returns a pandas dataframe
    with columns from the flattened objects
    """
    result = [
        MessageToDict(
            message,
            preserving_proto_field_name=True
        ) if message is not None else None
        for message in messages
    ]
    return pd.json_normalize(result)

def RSPClustersToMeasurments(measurement):
    """
    returns range, azimuth0, azimuth1, VrelRad and RCS from a measurement
    all are (N,)
    """
    if measurement is None:
        return None
    azimuth0 = np.array([cluster.a_AzAng[0] for cluster in measurement.a_RSPClusters])
    azimuth1 = np.array([cluster.a_AzAng[1] for cluster in measurement.a_RSPClusters])
    r = np.array([cluster.f_RangeRad for cluster in measurement.a_RSPClusters])
    VrelRad = np.array([cluster.f_VrelRad  for cluster in measurement.a_RSPClusters])
    RCS = np.array([cluster.f_RcsRaw   for cluster in measurement.a_RSPClusters])
    
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
    return points, azimuth1, VrelRad, RCS

class Transforms(object):

    """ Transformation matrix to CV2 Coordinates """
    TCV2 = np.array([
        [0., 0., 1., 0.],
        [-1, 0., 0., 0.],
        [0., -1., 0., 0.],
        [0., 0., 0., 1.]
    ])

    @staticmethod
    def R_yaw(alpha):
        """ Rotation Matrix around z """
        return np.array([
            [np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0.],
            [np.sin(np.radians(alpha)),  np.cos(np.radians(alpha)), 0.],
            [0., 0., 1.]
        ])

    @staticmethod
    def R_pitch(beta):
        """ Rotation Matrix around y """
        return np.array([
            [ np.cos(np.radians(beta)), 0, np.sin(np.radians(beta))],
            [0., 1., 0.],
            [-np.sin(np.radians(beta)), 0., np.cos(np.radians(beta))]
        ])

    @staticmethod
    def R_roll(gamma):
        """ Rotation Matrix around x """
        return np.array([
            [1., 0., 0.],
            [0., np.cos(np.radians(gamma)), -np.sin(np.radians(gamma))],
            [0., np.sin(np.radians(gamma)),  np.cos(np.radians(gamma))]
        ])

    @staticmethod
    def TSensor(yaw, pitch, roll, t_lon, t_lat, t_vert):
        """ Transformation from Sensor to World System """
        return np.row_stack([
            np.column_stack([
                Transforms.R_yaw(yaw) @ Transforms.R_pitch(pitch) @ Transforms.R_roll(roll),
                np.array([t_lon, t_lat, t_vert])
            ]),
            np.array([0., 0., 0., 1.])
        ])
    
    @staticmethod
    def FromPose(pose):
        """ Return 2 Transformation Matrices from a pose object """
        T1 = Transforms.TSensor(pose.fRoll, pose.fPitch, pose.fYaw, pose.fTx, pose.fTy, pose.fTz)
        T2 = np.array([
            [pose.sTransform.fA00, pose.sTransform.fA01, pose.sTransform.fA02, pose.sTransform.fA03],
            [pose.sTransform.fA10, pose.sTransform.fA11, pose.sTransform.fA12, pose.sTransform.fA13],
            [pose.sTransform.fA20, pose.sTransform.fA21, pose.sTransform.fA22, pose.sTransform.fA23],
            [0., 0., 0., 1.]
        ])
        return T1, T2

class ParquetExtractor(object):
    def __init__(self, parquet_file, lazy=False):
        """ Initializes a Parquet Reader, setting laze=True parses all signals
            and takes time during initialization, otherwise, signals are parsed
            on first access and parse results cached
        """
        self._data = pd.read_parquet(parquet_file, engine='pyarrow')

        self._ImageData = self._data[
            ['Image_Timestamp', 'Image_FrameCounter', 'Video_Suffix', 'Video_FrameCounter']
        ]
        self._CameraMounting = self.get_CameraMounting() if not lazy else None
        self._CameraPoseCalibration = self.get_CameraPoseCalibration() if not lazy else None
        self._CameraPoseDynamic = self.get_CameraPoseDynamic() if not lazy else None
        self._GPS = self.get_GPS() if not lazy else None
        self._GPSAsDataFrame = self.get_GPSAsDataFrame() if not lazy else None
        self._RSPClusters = self.get_RSPClusters() if not lazy else None
        self._VehDyn = self.get_VehDyn() if not lazy else None
        self._VehDynAsDataFrame = self.get_VehDynAsDataFrame() if not lazy else None
        self._VehPar = self.get_VehPar() if not lazy else None

        self._JoinedDataFrame = self.get_JoinedDataFrame() if not lazy else None

    @property
    def CameraMounting(self):
        """ a cached dictionary with vehicle parameters """
        if self._CameraMounting is None:
            self._CameraMounting = self.get_CameraMounting()
        return self._CameraMounting

    @property
    def CameraPoseCalibration(self):
        """ a cached dictionary with pose objects, keyed by image timestamp """
        if self._CameraPoseCalibration is None:
            self._CameraPoseCalibration = self.get_CameraPoseCalibration()
        return self._CameraPoseCalibration

    @property
    def CameraPoseDynamic(self):
        """ a cached dictionary with pose objects, keyed by image timestamp """
        if self._CameraPoseDynamic is None:
            self._CameraPoseDynamic = self.get_CameraPoseDynamic()
        return self._CameraPoseDynamic

    @property
    def GPS(self):
        """ A cached pandas dataframe with the GPS data """
        if self._GPS is None:
            self._GPS = self.get_GPS()
        return self._GPS

    @property
    def GPSAsDataFrame(self):
        """ A cached pandas dataframe with the GPS data """
        if self._GPSAsDataFrame is None:
            self._GPSAsDataFrame = self.get_GPSAsDataFrame()
        return self._GPSAsDataFrame

    @property
    def ImageData(self):
        """ A pandas dataframe with the image data """
        return self._ImageData

    @property
    def RSPClusters(self):
        if self._RSPClusters is None:
            self._RSPClusters = self.get_RSPClusters()
        return self._RSPClusters

    @property
    def VehDyn(self):
        """ a cached dictionary with VehDyn objects, keyed by image timestamp """
        if self._VehDyn is None:
            self._VehDyn = self.get_VehDyn()
        return self._VehDyn
    
    @property
    def VehDynAsDataFrame(self):
        """ a cached pandas dataframe with vehicle Dynamics """
        if self._VehDynAsDataFrame is None:
            self._VehDynAsDataFrame = self.get_VehDynAsDataFrame()
        return self._VehDynAsDataFrame

    @property
    def VehPar(self):
        """ a cached dictionary with vehicle parameters """
        if self._VehPar is None:
            self._VehPar = self.get_VehPar()
        return self._VehPar
    
    @property
    def JoinedDataFrame(self):
        """ a cached pandas dataframe of all non-constant values of the recording """
        if self._JoinedDataFrame is None:
            self._JoinedDataFrame = self.get_JoinedDataFrame()
        return self._JoinedDataFrame

    def get_Rad2CamTransformation(self):

        # Camera Transform from CameraMounting
        Tcam = Transforms.TSensor(
            #0, 29.5, 0, 
            0, 20, 0, 
            self.CameraMounting.get('LongPos', 1.8),
            self.CameraMounting.get('LatPos', 0.0),
            self.CameraMounting.get('VertPos', 1.67)
        )

        # Radar Transform from RadarMounting
        Trad = Transforms.TSensor(
            self.VehPar.get('sensorMounting.RollAngle', 0.0), 
            self.VehPar.get('sensorMounting.PitchAngle', 0.0),
            self.VehPar.get('sensorMounting.YawAngle', 0.0),
            self.VehPar.get('sensorMounting.LongPos', 1.86),
            self.VehPar.get('sensorMounting.LatPos', 0.0),
            self.VehPar.get('sensorMounting.VertPos', 0.62)
        )
        rad2cam = np.linalg.solve(Tcam @ Transforms.TCV2, Trad)
        return rad2cam

    def get_CameraMounting(self):
        """ A dictory with the camera position """
        MFC_SensorMounting = to_dataframe(deserialialize_protobuf(
            self._data.MFC5xx_Device_NVMAD_sRamDataFreeze_RAM_NVM_BLOCK_IPC_e_CL_IUC_VEHICLE_PARAMETERS_SensorMounting,
            SensorMounting_pb2.SensorMounting
        ))
        # if we don't have valid values at the end of the recording, there are none
        idx = -1
        return {
            feature: MFC_SensorMounting[feature].iloc[idx]
            for feature in MFC_SensorMounting.columns
        }

    def get_CameraPoseCalibration(self):
        """ a dictionary with camera poses indexed by image timestamp """
        MFC_SPoseCalibration = deserialialize_protobuf(
            self._data.MFC5xx_Device_ACAL_pMonoCalibration_SPoseCalibration,
            SPoseCalibration_pb2.SPoseCalibration
        )
        return {
            timestamp: pose
            for timestamp, pose in zip(self.ImageData.Image_Timestamp, MFC_SPoseCalibration)
        }

    def get_CameraPoseDynamic(self):
        """ a dictionary with camera poses indexed by image timestamp """
        MFC_SPoseDynamic =deserialialize_protobuf(
            self._data.MFC5xxDevice_ACAL_pMonoCalibration_SPoseDynamic,
            SPoseDynamic_pb2.SPoseDynamic
        )
        return {
            timestamp: pose
            for timestamp, pose in zip(self.ImageData.Image_Timestamp, MFC_SPoseDynamic)
        }

    def get_GPS(self):
        """ an ordered dictionary with GPS indexed by image timestamp """
        GPSDevice = deserialialize_protobuf(
            self._data.GPSDevice, GPSDevice_pb2.GPSDevice
        )
        return {
            timestamp: gps
            for timestamp, gps in zip(self.ImageData.Image_Timestamp, GPSDevice)
        }

    def get_GPSAsDataFrame(self):
        """ a pandas dataframe of GPS data index by image timestamp """
        GPS = to_dataframe(self.GPS.values())
        GPS.Valid = (GPS.Valid.values == '1').astype(int)
        return pd.concat(
            [self.ImageData, GPS], axis=1
        ).set_index('Image_Timestamp')

    def get_RSPClusters(self):
        """ a dict with image_timestamp as key and python objects describing radar measurements """
        radar = deserialialize_protobuf(
            self._data.ARS5xx_RawDataCycle_RSP2_ClusterListNS,
            RSP2_ClusterListNS_pb2.RSP2_ClusterListNS
        )
        return {
            timestamp: measurement
            for timestamp, measurement in zip(self.ImageData.Image_Timestamp, radar)
        }

    def get_VehDyn(self):
        VehDyn = deserialialize_protobuf(
            self._data.ARS5xx_AlgoVehCycle_VehDyn,
            VehDyn_pb2.VehDyn
        )
        return {
            ts: vd
            for ts, vd in zip(self.ImageData.Image_Timestamp, VehDyn)
        }

    def get_VehDynAsDataFrame(self):
        VehDyn = to_dataframe(self.VehDyn.values())
        VehDyn['Valid'] = (VehDyn['sSigHeader.eSigStatus'].values == '1').astype(int)
        return pd.concat(
            [self.ImageData, VehDyn],
            axis=1
        ).set_index('Image_Timestamp')

    def get_VehPar(self):
        """ A dictionary of Vehicle Parameters """
        ARS5xx_AlgoVehCycle_VehPar = to_dataframe(deserialialize_protobuf(
            self._data.ARS5xx_AlgoVehCycle_VehPar,
            VehPar_pb2.VehPar
        ))
        valid = ARS5xx_AlgoVehCycle_VehPar['sSigHeader.eSigStatus'] == '1'
        # if we don't have valid values at the end of the recording, there are none
        idx = -1
        return {
            feature: ARS5xx_AlgoVehCycle_VehPar[valid][feature].iloc[idx]
            for feature in ARS5xx_AlgoVehCycle_VehPar.columns
        }
    
    
    def get_JoinedDataFrame(self):
        CameraPoseCalibration = to_dataframe(self.CameraPoseCalibration.values()).add_prefix('CameraPoseCalibration.')
        CameraPoseDynamic = to_dataframe(self.CameraPoseDynamic.values()).add_prefix('CameraPoseDynamic.')
        GPS = to_dataframe(self.GPS.values()).add_prefix('GPS.')
        GPS['GPS.Valid'] = (GPS['GPS.Valid'].values == '1')
        VehDyn = to_dataframe(self.VehDyn.values()).add_prefix('VehDyn.')
        VehDyn['VehDyn.Valid'] = (VehDyn['VehDyn.sSigHeader.eSigStatus'].values == '1')

        joined = pd.concat(
            [self.ImageData, CameraPoseCalibration, CameraPoseDynamic, GPS, VehDyn],
            axis=1
        ).set_index('Image_Timestamp')
        joined[joined.select_dtypes(['object']).columns] = joined.select_dtypes(['object']).astype(str)
        return joined