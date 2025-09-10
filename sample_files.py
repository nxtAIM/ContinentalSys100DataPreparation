import glob
import os
import random
import subprocess

def sample():
    source_path = '/p/data1/nxtaim/proprietary/continental/sys100'
    target_path = '/p/scratch/nxtaim-1/proprietary/continental/sys100/raw'
    
    existing_recordings = [
        os.path.split(file)[-1]
        for file in glob.glob(os.path.join(target_path, 'sys100/*'))
    ]
    files = glob.glob(os.path.join(source_path, '*.tar'))
    random.shuffle(files)
    # print(existing_recordings)
    n = 0
    for file in files:
        rec_id, _ = os.path.splitext(os.path.split(file)[-1])
        if rec_id in existing_recordings:
            print(f'{file} already sampled')
            continue
        command = f'tar -xf {file} -C {target_path}'
        print(command)
        subprocess.run(command.split())
        n += 1
        if n >= 50:
            break

if __name__ == '__main__':
    sample()
