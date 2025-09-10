import os
import re
from pathlib import Path
import math


def partition_video_paths(
    root_dir,           # e.g., source directory
    output_base,        # e.g., target root directory
    shard_dir,          # where to write shard_000.txt, etc.
    num_shards=10,
    exts={'.mp4'},
    exclude_pattern=None,
):
    root_dir = Path(root_dir).resolve()
    output_base = Path(output_base).resolve()
    shard_dir = Path(shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    exclude_re = re.compile(exclude_pattern) if exclude_pattern else None

    all_videos = sorted([
        p for p in root_dir.rglob("*")
        if p.suffix in exts and (not exclude_re or not exclude_re.search(str(p)))
    ])

    print(f"Found {len(all_videos)} video files after applying filters.")

    for shard_id in range(num_shards):
        with open(shard_dir / f"shard_{shard_id:03d}.txt", "w") as f:
            for input_path in all_videos[shard_id::num_shards]:
                rel_path = input_path.relative_to(root_dir)
                output_dir = output_base / rel_path.parent
                f.write(f"{input_path}, {output_dir}\n")

    print(f"Partitioned into {num_shards} shards at {shard_dir}")


if __name__ == "__main__":
    partition_video_paths(
        root_dir="/p/data1/nxtaim/proprietary/continental/sys100",
        output_base="/p/data1/nxtaim/proprietary/continental/sys100_preprocessed",
        shard_dir="/p/data1/nxtaim/proprietary/continental/sys100_alternatives/partitions",
        num_shards=12,#24,
        exts={'.mp4'},
        exclude_pattern=r"gop\d+_.*crf\d+"
    )