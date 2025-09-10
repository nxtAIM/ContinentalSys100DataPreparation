import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import warnings

from enum import Enum

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "../Distortion"))
from distortion.mfc52x.cameras.camera_definitions import pinhole_params_by_levels
from distortion.mfc52x.cameras.camera_models import FisheyeCamera, PinholeCamera
from distortion.mfc52x.distortion import undistort_image, get_undistort_maps

def print_pinhole_cam_params(cam):
    if cam is not None:
        print(
            f"Pinhole camera paramters: fx={cam.fx}, fy={cam.fy}, \
            skew={cam.skew}, px={cam.px}, py={cam.py}, \
            image_width={cam.image_width}, image_height={cam.image_height}"
        )

def custom_warning(message, category, filename, lineno, file=None, line=None):
    print(f"ImageProcessingWarning: {message}")
warnings.showwarning = custom_warning


class ImageFormat(Enum):
    MFC510      = 510
    MFC52x_IMTA = 521
    MFC52x_MIPI = 522

def get_video_properties(cap, selection=['CAP_PROP_FRAME_WIDTH', 'CAP_PROP_FRAME_HEIGHT', 'CAP_PROP_FPS', 'CAP_PROP_FRAME_COUNT']):
    """
    Returns a dict of selected video properties,
    pass selection=None to return all available video properties
    """
    prefix = 'CAP_PROP_'
    properties = {
        k[len(prefix):] : cap.get(v)
        for k, v in cv2.__dict__.items() if k.startswith(prefix) and (selection is None or k in selection)
    }
    if ('FPS' in properties and 'FRAME_COUNT' in properties):
        properties['LENGTH_S'] = properties['FRAME_COUNT'] / properties['FPS']
    return properties

def get_format_from_bytearray(input_file_bytearray):
    """
    maps the size of a byte array to one of the supported formats
    return format, offset, width, height of the image
    """
    image_size = len(input_file_bytearray)
    if image_size == 1136016:
        #IMAGE_FORMAT_MFC510
        format = ImageFormat.MFC510
        offset = 3528
        width, height = (1176, 640)
    elif image_size == 2610560:
        #IMAGE_FORMAT_MFC52x_IMTA
        format = ImageFormat.MFC52x_IMTA
        offset = 5660
        width, height = (1828, 948)
    elif image_size == 2604672:
        #IMAGE_FORMAT_MFC52x_MIPI
        format = ImageFormat.MFC52x_MIPI
        offset = 5472
        width, height = (1824, 948)
    else:
        raise ValueError(f"no known encoding of the bytestream of length {image_size}")
    return format, offset, width, height

def get_format_from_image_size(width, height):
    if (width, height) == (1176, 640):
        return ImageFormat.MFC510
    elif (width, height) == (1824, 948):
        return ImageFormat.MFC52x_MIPI
    elif (width, height) == (1828, 948):
        return ImageFormat.MFC52x_IMTA
    else:
        raise ValueError(f"no known format for image of shape {width}x{height}")

def project_points(image, points, sizes, colors, K):
    """
    project points [[x,y,z]] onto image as circles of size in given colors
    use camera matrix K
    """
    image_points, _ = cv2.projectPoints(
        objectPoints=points,
        rvec=np.zeros(3),
        tvec=np.zeros(3),
        cameraMatrix=K,
        distCoeffs=np.zeros(5) # distortion Coefficients
    )
    for point, color, size in zip(image_points, colors, sizes):
        if point[0, 0] < 0 or point[0, 0] > image.shape[1] or point[0, 1] < 0 or point[0, 1] > image.shape[0]:
            continue
        image = cv2.circle(image, point[0].astype(int), int(size), [int(c) for c in color], -1)
    return image 

class FrameExtractor(object):
    """
    Extracts and undistorts individual frames from video files
    """
    known_video_types = ['.mp4']
    
    def __init__(self, source_video, image_path=None,
                 image_size=None, FOV=None, rel_center=None):
        """
        :param source_video: string to video location
        :param image_path: a string to the directory where the images should be saved at
        :param image_size: the target image size of the undistorted camera frames.
                           If None, default values from repository 'distortion' (3584, 1896)
        :param FOV: the target field of vision of the undistorted camera frames in radians,
                    If None, chooses the FOV of default values from repository 'distortion' (~110°)
        :param rel_center: where the center of the fisheye lense should be mapped to relative to
                           the width and height of the new undistored image, defaults to values (0.5, ~0.66)
        """

        self.cap = cv2.VideoCapture(source_video)
        self._video_properties = get_video_properties(self.cap)

        #self.mergemertens = mergemertens

        # not cropping currently
        #self.crop_top = 305
        #self.crop_height = 1134

        image_width = int(self.video_properties['FRAME_WIDTH'])
        image_height = int(self.video_properties['FRAME_HEIGHT'])

        self.image_format = get_format_from_image_size(
            image_width, image_height
        )
        self.is_flipped = self.image_format in [ImageFormat.MFC52x_IMTA, ImageFormat.MFC52x_MIPI]

        default = pinhole_params_by_levels['p0y']
        if image_size is None:
            image_size = (default['image_width'], default['image_height'])
        if FOV is None:
            f = default['fx']/default['image_width'] 
        else:
            f = 1 / 2 / np.tan(np.radians(FOV))
        if rel_center is None:
            c = (default['cx']/default['image_width'], default['cy']/default['image_height'])
        else:
            c = rel_center

        self.pinhole_cam_params = {
            'skew': 0,
            'image_width': image_size[0],
            'image_height': image_size[1],
            'cx': int(image_size[0] * c[0] + 0.5), 'cy': int(image_size[1] * c[1] + 0.5),
            'fx': int(image_size[0] * f + 0.5),    'fy': int(image_size[0] * f + 0.5)
        }
        self.fisheye_cam = FisheyeCamera(None, raw_width=image_width)
        self.pinhole_cam = PinholeCamera(self.pinhole_cam_params)
        print_pinhole_cam_params(self.pinhole_cam)

        self.map_x, self.map_y = get_undistort_maps(
            pinhole_cam_in=self.pinhole_cam, 
            fisheye_cam=self.fisheye_cam,
            is_flipped=self.is_flipped
        )

    def __del__(self):
        """ Destructor """
        self.cap.release()

    @property
    def pinhole_camera_matrix(self):
        return np.array([
            [self.pinhole_cam_params['fx'], 0., self.pinhole_cam_params['cx']],
            [0., self.pinhole_cam_params['fy'], self.pinhole_cam_params['cy']],
            [0., 0., 1.]
        ])

    @property
    def video_properties(self):
        return self._video_properties

    def extract_all_frames(self, outpath, limit=-1):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_counter = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            rgb = frame_to_rgb(frame)
            frame_counter += 1
            if limit > 0 and frame_counter > limit:
                break

    def get_raw_frame(self, framecounter):
        """ Get frame #framecounter from video - the first frame is framecounter = 1"""
        if framecounter < 1 or framecounter > self.video_properties['FRAME_COUNT']:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, framecounter - 1)
        ok, frame = self.cap.read()
        return frame if ok else None

    def undistort_frame(self, frame):

        frame = undistort_image(
            frame, self.fisheye_cam, self.pinhole_cam,
            map_x=self.map_x, map_y=self.map_y, is_flipped=self.is_flipped
        )

        return frame
