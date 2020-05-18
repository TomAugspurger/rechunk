import argparse
import ast
import os
import tempfile
from timeit import default_timer as tic

import dask
import dask.array as da
from distributed import Client, performance_report, wait
import rechunk


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original", default="original", help="File name of original store."
    )
    parser.add_argument("--split", default="split", help="File name of split store.")
    parser.add_argument("--final", default="final", help="File name of final store.")

    parser.add_argument(
        "-n", "--n-slices", default=4, type=int, help="Number of (time) slices."
    )
    parser.add_argument(
        "--shape",
        default=(4000, 4000),
        type=ast.literal_eval,
        help="Shape of other dimensions.",
    )
    parser.add_argument(
        "--split-chunks",
        default=None,
        type=ast.literal_eval,
        help="Split chunksize for rechunk.",
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args()

    ctx = directory = tempfile.TemporaryDirectory()

    with ctx:
        original = os.path.join(str(directory), args.original)
        split = os.path.join(str(directory), args.split)
        final = os.path.join(str(directory), args.final)

        shape = (args.n_slices,) + args.shape
        chunks = (1,) + args.shape
        a = da.random.random(shape, chunks=chunks)
        a.to_zarr(original, overwrite=True)

        with Client():
            print("rechunking")
            t0 = tic()

            with performance_report():
                rechunk.rechunk(original, split, final, args.split_chunks)
                t1 = tic()

        took = t1 - t0
        gbs = a.nbytes / 1e9 / took
        print(
            f"Rechunked {dask.utils.format_bytes(a.nbytes)} in {t1 - t0:.2f}s ({gbs:0.2f} GB/s)"
        )


if __name__ == "__main__":
    main()
