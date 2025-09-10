#!/usr/bin/bash

OUTPATH="/p/scratch/nxtaim-1/proprietary/continental/sys100"
file_prefixes=$(ls "$OUTPATH" | grep -oP '^\d{4}\.\d{2}\.\d{2}_at_\d{2}\.\d{2}\.\d{2}' | sort | uniq)

#echo $file_prefixes

N=10

missing_files=()

for input_dir in /p/scratch/nxtaim-1/proprietary/continental/sys100/raw/sys100_tar/*; do
    if [[ -d "$input_dir" ]]; then
	input_dir_name=$(basename "$input_dir")
	#echo $input_dir
	#echo $input_dir_name
	if ! echo "$file_prefixes" | grep -q "^$input_dir_name"; then
	    missing_files+=("$input_dir")
	fi
    fi
done

for input_dir in "${missing_files[@]}"; do
    echo "$input_dir"
    nice python ./extract_radar_frames.py "$input_dir" "$OUTPATH" &
    (( ++count % N == 0)) && wait
done
