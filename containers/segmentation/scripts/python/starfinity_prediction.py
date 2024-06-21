#!/usr/bin/env python

import os
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist3D
import argparse
import z5py
import dask
from dask.distributed import LocalCluster

def read_image(input_path):
    print("Reading image...")
    return imread(input_path)

def normalize_image(img):
    print("Normalizing image...")
    return normalize(img, 1, 99.8)

def predict(model, img, big=False, ignore_z=True, n_workers=4, batch_size=4,
            threads_per_worker=64, memory_limit='235GB', block_size=None):
    print("Weights loaded successfully")
    print("predicting...")
    if big:
        dask.config.set({
            'distributed.comm.max-frame-size': '1024MiB',
            'distributed.comm.timeouts.connect': '3600s',
            'distributed.comm.timeouts.tcp': '3600s',
            'distributed.scheduler.worker-ttl': None
        })


        threads_str = str(threads_per_worker - 2)
        os.environ['OMP_NUM_THREADS'] = threads_str
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            timeout="6000s"
        )
        if block_size is None:
            print("Warning: Please define block_size")

        n_tiles = tuple(int(np.ceil(s / 128)) for s in block_size)
        label_starfinity, res = model.predict_instances_big_dask(
            cluster, batch_size, img, axes='ZYX', block_size=block_size, min_overlap=[100, 100, 100],
            task_per_worker=batch_size // n_workers, ignore_z=ignore_z, context=None, n_tiles=n_tiles, affinity=True,
            affinity_thresh=0.1, verbose=True
        )
    else:
        img_normed = normalize_image(img)
        n_tiles = tuple(int(np.ceil(s / 128)) for s in img_normed.shape)
        time_usage, label_starfinity, res = model.predict_instances(img_normed, n_tiles=n_tiles,
                                                                    affinity=True, affinity_thresh=0.1,
                                                                    verbose=True)
        print(time_usage)

    return label_starfinity, res["markers"]

def save_results(outdir, label_starfinity):
    if outdir is not None:
        print("Saving results...")
        Path(outdir).mkdir(exist_ok=True, parents=True)
    imwrite(outdir, label_starfinity[:, np.newaxis].astype(np.uint16), imagej=True, compression='zlib')

def main(img, model, outdir, big=False, ignore_z=True, n_workers=4, batch_size=4, threads_per_worker=64, memory_limit='235GB', block_size=None):
    print("Image shape: ", img.shape)
    label_starfinity, label_stardist = predict(model, img, big=big, ignore_z=ignore_z,
                                               n_workers=n_workers, batch_size=batch_size,
                                               threads_per_worker=threads_per_worker, memory_limit=memory_limit,
                                               block_size=block_size)
    save_results(outdir, label_starfinity)
    print("Done")

def parse_num_blocks(num_blocks_str):
    return list(map(int, num_blocks_str.split()))

def recommended_parameters(model, img, required_block_num, ignore_z=True):
    print("Weights loaded successfully")
    block_size = model.check_blocksize(img, axes='ZYX', required_block_num=required_block_num, min_overlap=[100, 100, 100],
                          ignore_z=ignore_z, context=None)
    return block_size

def str2bool(value):
    if isinstance(value, bool):
       return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run StarDist3D prediction on a given image')
    parser.add_argument('-i', '--input', type=str, help="z5 input directory", required=True)
    parser.add_argument('-m', '--model', type=str, help="model directory", required=True)
    parser.add_argument('-o', '--output', type=str, help="output file", required=True)
    parser.add_argument('-c', '--channel', type=str, help="channel", required=True)
    parser.add_argument('-s', '--scale', type=str, help="scale", required=True)
    parser.add_argument('--big', type=str2bool, nargs='?', const=True, default=False, help="Enable big image processing with Dask (true/false)")
    parser.add_argument('--n_workers', type=int, default=4, help="Number of Dask workers")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for Dask processing")
    parser.add_argument('--threads_per_worker', type=int, default=64, help="Threads per Dask worker")
    parser.add_argument('--num_blocks', type=str, default='1 2 2', help="Block size for Dask processing")
    parser.add_argument('--memory_per_worker', type=str, default='235GB', help="Block size for Dask processing")

    args = parser.parse_args()

    # Convert num_blocks string to list of integers
    num_blocks = parse_num_blocks(args.num_blocks)
    if num_blocks[0] == 1:
        ignore_z = True
    else:
        ignore_z = False

    print("reading ...", args.input, args.channel + '/' + args.scale)
    im = z5py.File(args.input, use_zarr_format=False)
    img = im[args.channel + '/' + args.scale][:, :, :]
    model = StarDist3D(None, name=args.model, basedir='.')

    block_size = recommended_parameters(model, img, required_block_num=num_blocks, ignore_z=ignore_z)
    main(img, model, args.output, args.big, ignore_z, args.n_workers, args.batch_size,
         args.threads_per_worker, args.memory_per_worker, block_size)
