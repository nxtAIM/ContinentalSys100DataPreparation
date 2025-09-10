import sys
import os
import subprocess
import time
import socket
import multiprocessing as mp
from pathlib import Path
from reencoder import reencode_video

def format_seconds_date(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
def format_seconds(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def get_available_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else mp.cpu_count()

def process_video(input_path, output_path, delete_original=False):
    try:
        reencode_video(
            input_path=input_path,
            output_path=output_path,
            output_path_is_dir=True,
            gop_size=2,
            crf=16,
            preset='veryfast',
            recompile=True,
            debug=False,
            lossless=False,
            progress_updates=10,  # every 10 seconds or True for default 10s
            debug_id = os.environ.get("SLURM_PROCID", None),
        )
        if delete_original:
            print(f"deleting '{input_path}'",flush=True)
            input_path.unlink()
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed for '{input_path}' to '{output_path}':\n{e.stderr.decode()}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}", flush=True)
    return False

def main(shard_file, delete_original=False, max_workers=None):
    video_tasks = []
    with open(shard_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            input_str, output_str = map(str.strip, line.strip().split(",", maxsplit=1))
            video_tasks.append((Path(input_str), Path(output_str)))
    
    len_video_tasks = len(video_tasks)
    print(f"Node {socket.gethostname()} processing {len_video_tasks} videos from {shard_file}")

    num_workers = get_available_cpus()
    if max_workers is not None:
        num_workers = min(num_workers, max_workers)
    counter = mp.Value('i', 0)
    counter_lock = mp.Lock()
    debug_id = os.environ.get("SLURM_PROCID", None)
    debug_id = f"{debug_id:>3} " if debug_id is not None else ""
    start_time = time.time()

    def callback_done(success):
        with counter_lock:
            counter.value += 1
            status = "OK" if success else "FAIL"
            curr_time = time.time()
            print(f"[{debug_id}{format_seconds_date(curr_time)}] "
                 +f"{status} Progress {counter.value}/{len_video_tasks}  "
                 +f"[{format_seconds(curr_time-start_time)} < {format_seconds((curr_time-start_time)/counter.value*len_video_tasks)} Estimated Total] ",
                 flush=True)

    def error_callback(e):
        print(f"[{debug_id}ERROR CALLBACK]: {e}", flush=True)

    if num_workers == 1:
        # Serial execution
        for i, (input_path, output_path) in enumerate(video_tasks, start=1):
            try:
                success = process_video(input_path, output_path, delete_original)
                callback_done(success)
            except Exception as e:
                callback_done(False)
                error_callback(e)
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for input_path, output_path in video_tasks:
                r = pool.apply_async(
                    process_video,
                    args=(input_path, output_path, delete_original),
                    callback=callback_done,
                    error_callback=error_callback
                )
                results.append(r)

            for r in results:
                r.wait()
            pool.close()
            pool.join()
        
    print(f"Completed all tasks from {shard_file}")

def get_shard_path(shard_input: str | None) -> Path:
    script_dir = Path(__file__).resolve().parent
    slurm_rank = int(os.environ.get("SLURM_PROCID", 0))  # fallback to 0 if not in SLURM

    if shard_input is None:
        # Use default pattern if no input provided
        shard_dir = script_dir / "partitions"
        shard_file = shard_dir / f"shard_{slurm_rank:03d}.txt"
    else:
        input_path = Path(shard_input)
        if not input_path.is_absolute():
            input_path = (script_dir / input_path).resolve()
        
        if input_path.is_dir():
            # If input is a directory, look for shard file inside
            shard_file = input_path / f"shard_{slurm_rank:03d}.txt"
        else:
            shard_file = input_path

    if not shard_file.exists():
        raise FileNotFoundError(f"Shard file not found: {shard_file}")

    return shard_file

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch video reencoder")
    parser.add_argument(
        "--shard_file",
        type=str,
        default=None,
        help="absolute or relative Path to a shard file, or a directory containing shard files. "
             "If not provided, defaults to './partitions/shardsXXX.txt' based on SLURM_PROCID"
    )
    parser.add_argument("--delete", action="store_true", help="Delete original videos after encoding")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers (default: only one as ffmpeg already parallelises with as many as sensible)")

    args = parser.parse_args()

    shard_path = get_shard_path(args.shard_file)
    print(f"Max workers: {args.max_workers}, delete original video: {args.delete}", flush=True)
    print(f"Shard file: '{shard_path}'", flush=True)

    main(
        shard_file=shard_path,
        delete_original=args.delete,
        max_workers=args.max_workers
    )