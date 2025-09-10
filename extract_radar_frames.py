import argparse
import glob
import numpy as np
import os
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import re
from tqdm.auto import tqdm
import traceback

import frameExtractor as FE
import parquetExtractor as PE
import radarDiscretizer as RD


dtype_mapping = {
    "int64": pa.int64(),
    "int32": pa.int32(),
    "float64": pa.float64(),
    "float32": pa.float32(),
    "bool": pa.bool_(),
    "object": pa.string(),  # Assumes all objects are strings
    "string": pa.string(),  # Pandas 'string' dtype
    "datetime64[ns]": pa.timestamp("ns")
}

def make_pa_schema(radar_frame_example, image_frame_example, pe=None):
    """
    Generate a PyArrow schema from example radar and image frames
    """
    radar_frame_shape = radar_frame_example.shape
    radar_frame_dtype = radar_frame_example.dtype
    image_frame_shape = image_frame_example.shape
    image_frame_dtype = image_frame_example.dtype
    
    dtypes = [
        ("timestamp", pa.int64()),
        ("radar_frame", pa.binary()),
        ("image_frame", pa.binary())
    ]
    metadata={
        'radar_frame_shape': str(radar_frame_shape),
        'radar_frame_dtype': str(radar_frame_dtype),
        'image_frame_shape': str(image_frame_shape),
        'image_frame_dtype': str(image_frame_dtype)
    }
    if pe is not None:
        dtypes += [
            (col, dtype_mapping[str(dtype)]) for col, dtype in pe.JoinedDataFrame.dtypes.items()
        ]
        metadata.update({
            f"CameraMounting.{k}": str(v) for k,v in pe.CameraMounting.items()
        })
        metadata.update({
            f"VehPar.{k}": str(v) for k,v in pe.VehPar.items()
        })
        Rad2CamTransformation = pe.get_Rad2CamTransformation()
        metadata.update({
            f"Rad2CamTransformation.A{i}{j}": f"{Rad2CamTransformation[i,j]:.8f}"
            for j in range(4) for i in range(3)
        })

    schema = pa.schema(dtypes, metadata=metadata)
    return schema
    
def read_from_parquet(filename, batch_size=1):
    """
    Reads a Parquet file and returns a generator of timestamps, radar and image frames.

    Parameters:
    - file_name: str, name of the Parquet file to read the data from

    Returns generator of :
    - timestamps: list of np.int64
    - radar frames: list of np.ndarray (each of original shape and dtype)
    - image frames: list of np.ndarray (each of original shape and dtype)
    """
    # Extract metadata
    metadata = pq.read_metadata(filename)
    metadata = metadata.schema.to_arrow_schema().metadata
    radar_frame_shape = eval(metadata[b'radar_frame_shape'].decode('utf-8'))
    radar_frame_dtype = np.dtype(metadata[b'radar_frame_dtype'].decode('utf-8'))
    image_frame_shape = eval(metadata[b'image_frame_shape'].decode('utf-8'))
    image_frame_dtype = np.dtype(metadata[b'image_frame_dtype'].decode('utf-8'))
    
    # Read the Parquet file
    parquet_file = pq.ParquetFile(filename)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        
        # Extract data
        timestamps = batch.column('timestamp').to_pylist()
        binary_radar_frames = batch.column('radar_frame').to_pylist()
        binary_image_frames = batch.column('image_frame').to_pylist()

        # Convert binary frames back to numpy arrays
        radar_frames = [
            np.frombuffer(frame, dtype=radar_frame_dtype).reshape(radar_frame_shape)
            for frame in binary_radar_frames
        ]
        image_frames = [
            np.frombuffer(frame, dtype=image_frame_dtype).reshape(image_frame_shape)
            for frame in binary_image_frames
        ]

        yield timestamps, radar_frames, image_frames

def make_argument_parser():
    parser = argparse.ArgumentParser(
        prog='extract_radar_frames.py',
        description='Extract Radar data as images from recording data',
        epilog='Have fun extrating!'
    )

    parser.add_argument('inputdir')
    parser.add_argument('outputdir')
    parser.add_argument('-s', '--discretization_steps', type=int, default=64)
    parser.add_argument('-v', '--valid_indicator', type=int, default=1)
    parser.add_argument('-xmin', type=int, default=0)
    parser.add_argument('-xmax', type=int, default=102)
    parser.add_argument('-ymin', type=int, default=-100)
    parser.add_argument('-ymax', type=int, default=100)

    return parser

def write_radar_image_parquet(outfile, pe, fe_odd, fe_even, discretizer, pa_schema, batch_size=1):
    
    timestamp_batch = []
    radar_frame_batch = []
    image_frame_batch = []

    schema_checked = False

    with pq.ParquetWriter(outfile, schema=pa_schema) as writer:
        for idx, row in tqdm(pe.ImageData.iterrows(), total=pe.ImageData.shape[0]):

            if row.Video_Suffix == '_ODD':
                ok, image_frame = fe_odd.cap.read()
            elif row.Video_Suffix == '_EVEN':
                ok, image_frame = fe_even.cap.read()
            else:
                continue
            if not ok:
                continue

            # brg to rgb
            image_frame = image_frame[...,::-1]

            RSPClusters = pe.RSPClusters[row.Image_Timestamp]
            if RSPClusters is None:
                continue

            points, azimuth1, vel, RCS = PE.RSPClustersToPoints(RSPClusters)

            # put the rangeGateLength into the data and adjust the discretizer
            rangeGateLength = RSPClusters.clustListHead.f_RangeGateLength
            discretizer._xmax = rangeGateLength * 256
            points[-1, :] = rangeGateLength
            
            grid = discretizer.to_grid(np.vstack([points, azimuth1, vel, RCS]).T)
            radar_frame = discretizer.grid_to_image(grid, swap_xy=False, invert_rows=True, invert_columns=True).astype(np.float32)

            if not schema_checked:
                assert eval(pa_schema.metadata[b'radar_frame_shape']) == radar_frame.shape
                assert np.dtype(pa_schema.metadata[b'radar_frame_dtype']) == radar_frame.dtype
                assert eval(pa_schema.metadata[b'image_frame_shape']) == image_frame.shape
                assert np.dtype(pa_schema.metadata[b'image_frame_dtype']) == image_frame.dtype
                schema_checked = True

            timestamp_batch.append(row.Image_Timestamp)
            image_frame_batch.append(image_frame[:, :, :].tobytes())
            radar_frame_batch.append(radar_frame.tobytes())

            if len(image_frame_batch) % batch_size == 0:
                # https://stackoverflow.com/questions/64791558/create-parquet-files-from-stream-in-python-in-memory-efficient-manner
                subdf = pe.JoinedDataFrame.loc[timestamp_batch]
                batch = pa.RecordBatch.from_arrays([
                    pa.array(timestamp_batch, type=pa_schema.field('timestamp').type),
                    pa.array(radar_frame_batch, type=pa_schema.field('radar_frame').type),
                    pa.array(image_frame_batch, type=pa_schema.field('image_frame').type)
                    ]+[
                    pa.array(subdf[col].tolist(), type=pa_schema.field(col).type)
                    for col in subdf.columns
                    ],
                    schema=pa_schema
                )
                writer.write_batch(batch)
                timestamp_batch.clear()
                radar_frame_batch.clear()
                image_frame_batch.clear()

def main(args):
    regexp = f'^(\d{{4}}\.\d{{2}}\.\d{{2}}_at_\d{{2}}\.\d{{2}}\.\d{{2}}_camera-radar-\w{{2}}_\d{{4}}).parquet$'
    
    source_files = glob.glob(os.path.join(args.inputdir, '*.parquet'))
    pathlib.Path(args.outputdir).mkdir(parents=True, exist_ok=True)
    
    discretizer = RD.RadarDiscretizer(
        xmin=args.xmin, xmax=args.xmax, 
        ymin=args.ymin, ymax=args.ymax, 
        discretization_steps=args.discretization_steps,
        valid_indicator=args.valid_indicator
    )
    
    for parquet_file in source_files:
        try:
            
            path, filename = os.path.split(parquet_file)
            if not re.match(regexp, filename):
                print(f'{filename} does not match expected name pattern, skipping')
                continue
                
            base_name, ext = os.path.splitext(parquet_file)
            odd_video_file = f'{base_name}_ODD.mp4'
            even_video_file = f'{base_name}_EVEN.mp4'
            
            if not os.path.isfile(odd_video_file):
                print(f'No Odd video file found for {parquet_file}, skipping')
                continue
            if not os.path.isfile(even_video_file):
                print(f'No Even video file found for {parquet_file}, skipping')
                continue

            pe = PE.ParquetExtractor(parquet_file, lazy=True)
            fe_odd = FE.FrameExtractor(odd_video_file)
            fe_even = FE.FrameExtractor(even_video_file)
            
            total_video_frames = fe_odd.video_properties['FRAME_COUNT'] + fe_even.video_properties['FRAME_COUNT']
            total_radar_frames = pe.ImageData.shape[0]
            if not total_video_frames == total_radar_frames:
                print(f'Video and radar Frames don not match in {parquet_file}, skipping')
                continue
            
            pa_schema = make_pa_schema(
                # Attention!!, the feature dimension is hard-coded here: x, y, z, azimuth_1, vrel, RCD, valid = 7
                np.zeros((args.discretization_steps, args.discretization_steps, 7), dtype=np.float32),
                np.zeros((int(fe_odd.video_properties['FRAME_HEIGHT']), int(fe_odd.video_properties['FRAME_WIDTH']), 3), dtype=np.uint8),
                pe=pe,
            )
            
            basename, ext = os.path.splitext(filename)
            target_file = os.path.join(args.outputdir, f'{filename}_radargrids_{args.discretization_steps}.parquet')
            print(f'Extracing {parquet_file} to {target_file}')

            write_radar_image_parquet(target_file, pe, fe_odd, fe_even, discretizer, pa_schema)

            print(f'Wrote {target_file}.')
            
        except Exception as e:
            print(e)
            print(f'Unknown Problmes with {parquet_file} - skipping')
            print(f'>>> Traceback')
            traceback.print_exc()
        
    print('Done')

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    main(args)