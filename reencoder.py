import subprocess
from pathlib import Path
import argparse
import cv2
import av
import time

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 Failed to open video {video_path}")
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
def format_seconds(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def reencode_video(
    input_path,
    output_path=None,
    output_path_is_dir=False,
    progress_updates : bool|int =False,
    gop_size=15,
    crf=23,
    preset='veryfast',
    keyint_min=None,
    sc_threshold=None,
    lossless=False,
    recompile=True,
    debug=False,
    debug_id=None
):
    input_path = Path(input_path)
    
    suffix_parts = [f'gop{gop_size}']
    if lossless:
        suffix_parts.append('lossless')
    else:
        suffix_parts.append(f'crf{crf}')
    suffix_parts.append(preset)
    if keyint_min is not None:
        suffix_parts.append(f'keyintmin{keyint_min}')
    if sc_threshold is not None:
        suffix_parts.append(f'scthresh{sc_threshold}')
    suffix = '_'.join(suffix_parts)
    
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_{suffix}.mp4")
    else:
        output_path = Path(output_path)
        if output_path_is_dir or not output_path.suffix or output_path.is_dir():
        # Treat as directory if no suffix or if it already exists as a dir
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"{input_path.stem}_{suffix}.mp4"
    
    if output_path.exists() and not recompile:
        print(f"Output file {output_path} already exists. Skipping recompilation.")
        return output_path
    
    if debug:
        cmd = ['ffmpeg']
    else:
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',
        ]
    
    cmd += ['-i', str(input_path)]
    cmd += ['-c:v', 'libx264']
    cmd += ['-preset', preset]
    cmd += ['-g', str(gop_size)]
    cmd += ['-pix_fmt', 'yuv420p']

    if lossless:
        cmd += ['-qp', '0']  # lossless mode disables crf
    else:
        cmd += ['-crf', str(crf)]
    
    if keyint_min is not None:
        cmd += ['-keyint_min', str(keyint_min)]
    if sc_threshold is not None:
        cmd += ['-sc_threshold', str(sc_threshold)]
    
    cmd += ['-y', str(output_path)]

    if progress_updates is True:
        progress_updates = 10
    if isinstance(progress_updates, int) and progress_updates > 0:
        cmd += ['-progress', 'pipe:1', '-nostats']
        total_frames = get_total_frames(input_path)
        #print(f"Running: {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        last_report = time.perf_counter()
        start_time = last_report
        progress_data = {}
        debug_prefix = "" if debug_id is None else f"{str(debug_id):>3} "

        def parse_line(line):
            if '=' not in line:
                return None, None
            k, v = line.strip().split('=', 1)
            return k, v
        def parse_progress_line(now):
            frame = int(progress_data.get('frame', 0))
            needed_time = now - start_time
            estimated_time = needed_time * total_frames / max(frame, 1)
            stem = str(input_path.stem)
            print(f"[{debug_prefix}Progress {stem[:22]+stem[-4:]:>26}] "
                    +f"frame={frame:>5}/{total_frames:>5} "
                    +f"speed={progress_data.get('speed', 'N/A'):>6} "
                    +f"duplicated={progress_data.get('dup_frames', 'N/A'):>3} "
                    +f"dropped={progress_data.get('drop_frames', 'N/A'):>3} "
                    +f"[{format_seconds(needed_time)} < {format_seconds(estimated_time)}]",
                flush = True
            )

        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            key, value = parse_line(line)
            if key:
                progress_data[key] = value

            now = time.perf_counter()
            if now - last_report > progress_updates:
                # Example progress print
                parse_progress_line(now)
                last_report = now

            if progress_data.get('progress') == 'end':
                parse_progress_line(now)
                break
        # Wait for process to finish and get exit code
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        #print(f"Running: {' '.join(cmd)}\n")
        if debug:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    #print(f"\nOutput saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Re-encode video with custom keyframe settings for better random access.')
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('--output', type=str, default=None, help='Output video file path (optional)')
    parser.add_argument('--gop_size', type=int, default=15, help='Keyframe interval (GOP size), default=15')
    parser.add_argument('--crf', type=int, default=23, help='CRF quality setting, default=23')
    parser.add_argument('--preset', type=str, default='veryfast', help='Encoding preset, default=veryfast')
    parser.add_argument('--keyint_min', type=int, default=None, help='Minimum keyframe interval (optional)')
    parser.add_argument('--sc_threshold', type=int, default=None, help='Scene change threshold (optional)')

    args = parser.parse_args()

    reencode_video(
        args.input_video,
        output_path=args.output,
        gop_size=args.gop_size,
        crf=args.crf,
        preset=args.preset,
        keyint_min=args.keyint_min,
        sc_threshold=args.sc_threshold
    )

if __name__ == '__main__':
    #main()
    import sys, os
    from pathlib import Path
    import random

    vid_path = Path("/p/data1/nxtaim/proprietary/continental/sys100_alternatives/2021.04.15_at_11.38.16/2021.04.15_at_11.38.16_camera-radar-mi_5316_EVEN.mp4")
    recompile = False
    checkfor = 300

    kwarg_list = [
        {"input_path": vid_path, "gop_size": 10},
        {"input_path": vid_path, "gop_size": 5},
        {"input_path": vid_path, "gop_size": 2},
        {"input_path": vid_path, "gop_size": 10, "keyint_min":10},
        {"input_path": vid_path, "gop_size": 10, "preset":"medium"},
        {"input_path": vid_path, "gop_size": 10, "preset":"slower"},
        {"input_path": vid_path, "gop_size": 10, "lossless":True},
        {"input_path": vid_path, "gop_size": 5, "lossless":True},
        {"input_path": vid_path, "gop_size": 10, "crf":16},
        {"input_path": vid_path, "gop_size": 5, "crf":16},
        {"input_path": vid_path, "gop_size": 2, "crf":16, "debug":True},
    ]
    outpaths = []
    for kwargs in kwarg_list:
        kwargs["recompile"] = recompile
        start_time = time.perf_counter()
        outpaths.append(reencode_video(**kwargs))
        total_time = time.perf_counter() - start_time
        print(f"\n Re-enconcodeing took {format_seconds(total_time)}\n")
    
    outpaths.append(vid_path)
    basename = Path(vid_path).stem

    basepath = "/p/project1/nxtaim-1/neuhoefer1/CleanStableDiffusion"
    if not basepath in sys.path:
        sys.path.insert(0, basepath)

    from src.dataloading.videodataset import VideoDataset

    print("\nTesting random access to a **single** video, compiled in different formats")
    print("Note that random access speed will be lower when reading from a large"
         +" collection of video files as we do not get the benefit of accessing"
         +" the same area of the disk storage system and benefit from the"
         +" automatic buffering of stored chunks\n")


    for test_vid in outpaths:
        file_size = os.path.getsize(test_vid)
        cap = cv2.VideoCapture(filename=test_vid)
        if not cap.isOpened():
            raise ValueError(f"Video file '{test_vid}' could not be opened!")
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        print((Path(test_vid).stem).replace(basename, 'base'),":")
        print(f"{frames} frames at {fps} fps, {width} x {height}, {file_size/2**20:>8.2f} MB")
        for extract_method in [
            "ffmpeg",
            "cv2",
            "torchvision",
            "torchcodec",
            #"torchcodec_approximate",
            "decord",
            "pyav",
        ]:
            dataset = VideoDataset(
                source=test_vid,
                extract_function=extract_method,
                out_img_size=(512,512),
                quiet=True,
            )
            iter = 0
            needed_time = 0
            start_time = time.perf_counter()
            while needed_time < checkfor:
                x = dataset[random.randrange(len(dataset))]
                iter += 1
                needed_time = time.perf_counter() - start_time
            print(f"   {extract_method:>12}: {iter/needed_time:>6.2f} frames/sec  ({iter:>4d} frames in {needed_time:.2f}s)")
        print()
    print(f"Remeber to delete the differently encoded video copies in '{vid_path.parent}'")