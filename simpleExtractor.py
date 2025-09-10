from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import io
import time
import warnings
import subprocess
import threading
#from functools import partial
from enum import Enum

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "../Distortion"))
from distortion.mfc52x.cameras.camera_definitions import pinhole_params_by_levels
from distortion.mfc52x.cameras.camera_models import FisheyeCamera, PinholeCamera
from distortion.mfc52x.distortion import undistort_image, get_undistort_maps

from config import FFMPEG_EXEC



def print_pinhole_cam_params(cam):
    if cam is not None:
        print(f"Pinhole camera paramters: fx={cam.fx}, fy={cam.fy}, skew={cam.skew}, px={cam.px}, py={cam.py}, image_width={cam.image_width}, image_height={cam.image_height}")
def custom_warning(message, category, filename, lineno, file=None, line=None):
    print(f"ImageProcessingWarning: {message}")
warnings.showwarning = custom_warning


class ImageFormat(Enum):
    MFC510      = 510
    MFC52x_IMTA = 521
    MFC52x_MIPI = 522
def extract_bin(filepath):
    # The input file path should lead to .bin of binary data of 12-bit Bayer-GB camera images 
    # (Bayer-GB or GBRG is a format where each pixel in a 2x2 square only detects an individual colour in a [[G,B],[R,G]] scheme )
    # Since numpy only has a 8-bit and 16-bit uints we transform the data into 16-bit uint array
    # and read three consecutive 8-bit integers which carry two 12-bit data integers. 
    # In which pattern the bits are distributed differs by format
    # In particular, we use the 12 most significant bits of the 16 bits of space for the 12 bits of data
    # such that the interpolation scheme from GBRG to BGR (not RGB in preparation for cv2) can use the most precision available
    input_file_bytearray = np.fromfile(filepath, dtype=np.uint8)
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
    size = width*height
    size_limit = int(3 * size / 2) + offset
    arr = input_file_bytearray[offset:size_limit].astype(np.uint16)
    out = np.empty(size,dtype=np.uint16)
    if format == ImageFormat.MFC510:
        out[0::2] = (arr[0::3] <<  8) | ( arr[1::3] & 240)
        out[1::2] = (arr[1::3] << 12) | ((arr[2::3])      << 4)
    elif format == ImageFormat.MFC52x_IMTA:
        out[0::2] = (arr[1::3] << 12) | ( arr[0::3]       << 4)
        out[1::2] = (arr[2::3] <<  8) | ( arr[1::3] & 240)
    elif format == ImageFormat.MFC52x_MIPI:
        out[0::2] = (arr[0::3] <<  8) | ((arr[2::3] & 15) << 4)
        out[1::2] = (arr[1::3] <<  8) | ( arr[2::3] & 240)
    return format, (cv2.cvtColor(out.reshape((height,width)), cv2.COLOR_BAYER_GB2BGR) >> 8).astype(np.uint8)


class SimpleExtractor(object):

    known_image_types = ['.bin']
    
    def __init__(self, video_path,  #mergemertens=False,
                 image_size=None, FOV=None, rel_center=None, video_nbr=2,
                 fps=30, use_cv2_VideoWriter=True, codec_fourcc='mp4v', 
                 quality=23, lossless=False, buffered_images=4):
        # :param video_path: a string to the directory where the videos should be saved at
        # :param image_size: the target image size of the undistorted camera frames.
        #                    If None, default values from repository 'distortion' (3584, 1896)
        # :param FOV: the target field of vision of the undistorted camera frames in radians,
        #             If None, chooses the FOV of default values from repository 'distortion' (~110°)
        # :param rel_center: where the center of the fisheye lense should be mapped to relative to
        #                    the width and height of the new undistored image, defaults to values (0.5, ~0.66)
        # :param video_nbr: over how many videos the individual frames should be distributed,
        #                   note that every second frames has a different eposure time
        # :param fps: the frames per second the input frames are assumed to be recorded at
        # :param use_cv2_VideoWriter: whether to use cv2 to write the Video, if False,
        #                             creates 'video_nbr' many subprocesses using ffmpeg for video writing
        # :param codec_fourcc: the codec for the cv2 videoWriter, only has an effect if use_cv2_VideoWriter==True
        # :param quality: the quality parameter (int between 0 and 51) of ffmpeg video compression, lower values signify better quality,
        #                 only has an effect if use_cv2_VideoWriter==False and lossless==False
        # :param lossless: whether the compression should be lossless, only works if use_cv2_VideoWriter==False.
        #                  Sets the quantization parameter (-qp) to zero for every frame
        # :param buffered_images: how big the buffer for incoming images for each ffmpeg command should be in
        #                         relation to the individual image sizes, note that the provided buffer will 
        #                         always be a power of 2 and slightly larger
        self.video_path = video_path
        self.fps = fps/video_nbr
        self.framecounter = 0
        #self.mergemertens = mergemertens

        # not cropping currently
        #self.crop_top = 305
        #self.crop_height = 1134

        # TODO - Best Choice for Video Codec?
        self.codec = cv2.VideoWriter_fourcc(*codec_fourcc)
        
        self.use_cv2_VideoWriter = use_cv2_VideoWriter
        self.video_nbr = video_nbr
        self.video_writers = []
        self.crf = quality
        self.qp = lossless
        self.buffered_images = buffered_images
        
        self.image_format = None
        self.pinhole_cam = None
        self.fisheye_cam = None
        self.map_x = None
        self.map_y = None
        self.format = None
        self.is_flipped = False

        default = pinhole_params_by_levels['p0y']
        if image_size is None:
            image_size = (default['image_width'], default['image_height'])
        if FOV is None:
            f = default['fx']/default['image_width'] 
        else:
            f = 1/2/np.tan(FOV*np.pi/360)
        if rel_center is None:
            c = (default['cx']/default['image_width'], default['cy']/default['image_height'])
        else:
            c = rel_center
        self.pinhole_cam_params = { 'skew': 0,
            'image_width': image_size[0], 'image_height': image_size[1],
            'cx': int(image_size[0]*c[0]+0.5), 'cy': int(image_size[1]*c[1]+0.5),
            'fx': int(image_size[0]*f+0.5),    'fy': int(image_size[0]*f+0.5)
        }

    def init_videos(self, rgb):
        frameSize = (rgb.shape[1],rgb.shape[0])
        if self.video_nbr == 1:
            suffixs = ['ALL']
        elif self.video_nbr == 2:
            suffixs = ['EVEN', 'ODD']
        else:
            suffixs = [f'{i}Modulo{self.video_nbr}' for i in range(self.video_nbr)]
        #if self.mergemertens:
        #    self.MergeMertensProcessor = cv2.createMergeMertens()
        #    suffixs = ['MergeMertens']
        if self.use_cv2_VideoWriter:
            add_suffix = ""
        elif self.qp:
            add_suffix = "_lossless"
        else:
            add_suffix = f"_quality{self.crf}"
        
        if self.use_cv2_VideoWriter:
            print("Initialising cv2 VideoWriters")
            self.video_writers += [
                cv2.VideoWriter(
                    os.path.join(self.video_path, f'{suffix}{add_suffix}.mp4'),
                    self.codec, fps=self.fps, frameSize=frameSize
                )
                for suffix in suffixs
            ]
        else:
            # https://stackoverflow.com/a/60572878
            print("Initialising ffmpeg pipelines as subprocesses")
            buffsize = 2*int(np.log2(self.buffered_images*3*np.prod(frameSize))/2+1)
            print(f"  buffer: {buffsize}-bit")
            for suffix in suffixs:
                outfile = os.path.join(self.video_path, f'{suffix}{add_suffix}.mp4')
                print(" - output in", outfile)
                ffmpeg_command = [
                    FFMPEG_EXEC,
                    '-loglevel', 'error',
                    '-y',                                    # Overwrite output file if it exists
                    # Input params
                    '-f', 'rawvideo',                        # input data format is raw data byte stream
                    '-vcodec', 'rawvideo',                   # input data is not encoded
                    '-pix_fmt', 'bgr24',                     # our input images are bgr values with 3 8-bit values
                    '-s', f'{rgb.shape[1]}x{rgb.shape[0]}',  # input frame sizes are known so bytestream can be interpreted
                    '-r', str(self.fps),                     # the input frames have known fps
                    '-an', '-sn',                            # the input data stream does not contain audio or subtitle information
                    '-i', 'pipe:',                           # Input from stdin
                    # Output params
                    '-vcodec', 'libx264',                    # Output video codec (vp9 slower but better compressed)
                    '-pix_fmt', 'yuv420p',                   # Pixel format (yuv420p is supported everywhere and 420p is not related to resolution)
                    ('-qp' if self.qp else '-crf'),          # '-qp 0' (quantization parameter constantly 0) results in lossless compression otherwise 
                    ('0' if self.qp else str(self.crf)),     #   '-crf' (Constant Rate Factor) controls the change of the qp for dynamic frames 
                                                             #   crf can be a value between 0 and 51, lower means better quality but bigger file sizes
                    outfile,
                ]
                self.video_writers.append(
                    subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, #stderr=subprocess.PIPE,
                                     bufsize=2**buffsize)
                )
    
    def release_videos(self):
        if self.use_cv2_VideoWriter:
            for video_writer in self.video_writers:
                video_writer.release()
        else:
            for video_writer in self.video_writers:
                video_writer.stdin.close()
            if len(self.video_writers) > 0:
                print("waiting for ffmpeg to finish encoding")
            for video_writer in self.video_writers:
                video_writer.wait()
        video_writers = []
    
    def bin_img_to_rgb(self, filepath):
        format, rgb = extract_bin(filepath)
        if format != self.format:
            self.is_flipped = format in [ImageFormat.MFC52x_IMTA, ImageFormat.MFC52x_MIPI]
            self.format = format
        if self.fisheye_cam is None:
            self.fisheye_cam = FisheyeCamera(None, raw_width=rgb.shape[1])
        if self.pinhole_cam is None:
            # we should be able to define the output image shapes
            self.pinhole_cam = PinholeCamera(self.pinhole_cam_params)
            print_pinhole_cam_params(self.pinhole_cam)
            # we should also end previous videos if camera paramters have changed
            self.release_videos()
        if self.map_x is None or self.map_y is None:
            # in another version is_flipped should only be considered in undistord_image
            self.map_x, self.map_y = get_undistort_maps(pinhole_cam_in=self.pinhole_cam, 
                                                        fisheye_cam=self.fisheye_cam,
                                                        is_flipped=self.is_flipped) # is_flipped = False
        rgb = undistort_image(
                rgb, self.fisheye_cam, self.pinhole_cam,
                map_x=self.map_x, map_y=self.map_y, is_flipped=self.is_flipped)
        if not hasattr(rgb, "shape"):
            rgb = rgb[0] # there seems to be versions of undistort_image that also return the maps
        return rgb


    def add_image_to_video(self, image_path):

        image_extension = os.path.splitext(image_path)[1]
        if image_extension == '.bin':
            try:
                rgb = self.bin_img_to_rgb(image_path)
            except ValueError as e:
                warnings.warn("couldn't process image, it is hence ignored! The error measge was: \n  "+str(e))
                return 1
        else:
            raise ValueError(f"unkown image type '{image_extension}' in '{image_path}'"+
                             "\nknown extensions are ['.bin']")
        if len(self.video_writers) == 0:
            self.init_videos(rgb)

        #if self.mergemertens:
        #    if self.framecounter % 2 == 0:
        #        self.last_rgb = rgb
        #    else:
        #        fusion = self.MergeMertensProcessor.process([self.last_rgb, rgb])
        #        np.clip(fusion*255, 0, 255, out=fusion)
        #        self.video_writers[0].write(fusion.astype(np.uint8))
        #else:
        if self.use_cv2_VideoWriter:
            self.video_writers[self.framecounter % self.video_nbr].write(rgb)
        else:
            self.video_writers[self.framecounter % self.video_nbr].stdin.write(rgb.tobytes())
            # if input does not fit into the buffer, this process waits for the ffmpeg subprocesses

        self.framecounter += 1

        return 0