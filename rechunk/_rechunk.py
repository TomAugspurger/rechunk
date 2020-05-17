"""
TODO:

* Choose split chunksize
* Partial merges
* Choose merge axis
"""
import numpy as np
import zarr
from distributed import get_client, wait
from dask.bytes.core import get_mapper
import dask.array as da


def rechunk(original, split, final, split_chunks=None):
    """
    Rechunk a dataset.
    """
    a = da.from_zarr(original)
    chunks = {i: "auto" for i in range(a.ndim)}
    chunks[0] = -1
    if split_chunks is None:
        chunksize = a.rechunk(chunks).chunks[1][0]
        split_chunks = (chunksize,) * (a.ndim - 1)

    client = get_client()
    fs = client.map(split_and_store, list(range(a.numblocks[0])),
                    src=original, dst=split, split_chunks=split_chunks)
    wait(fs)

    n = np.prod(da.from_zarr(split).numblocks[1:])
    fs = client.map(merge_and_store, range(n), src=split, dst=final)
    wait(fs)
    return da.from_zarr(final)


def split_and_store(i, src, dst, split_chunks):
    """
    Split a Dask array along multiple dimensions and store the output.

    Parameters
    ----------
    i : int
        The block number of the dask Array
    src, dst : pathlike
        The files to read from and store to.
    split_chunks : tuple
        The new chunksize for the chunks you're splitting.
    """
    mapper = get_mapper(src)
    A = zarr.Array(mapper, read_only=True)
    slices = da.core.slices_from_chunks(da.core.normalize_chunks(A.chunks,
                                                                 A.shape))
    slice_ = slices[i]
    chunk = A[slice_]  # ndarray
    # chunks = (2000, 2000)
    chunks = split_chunks
    store = zarr.open(dst, mode="a", shape=A.shape, chunks=(1,) + chunks,
                      dtype=A.dtype)
    store

    assert store.shape == A.shape
    store[slice_] = chunk
    chunks2 = da.core.normalize_chunks((1,) + split_chunks, chunk.shape)
    slices2 = da.core.slices_from_chunks(chunks2)

    for slice2 in slices2:
        target = (slice(i, i + 1),) + slice2[1:]
        store[target] = chunk[slice2]


def merge_and_store(i, src, dst):
    """
    Merge a dimension of the Dask Array and store the results.

    Paramters
    ---------
    i : int
        The block number of the final array.
    src, dst: pathlike
        The files to read from and write to.
    """
    mapper = get_mapper(src)
    A = zarr.Array(mapper, read_only=True)
    zchunks = (A.shape[0],) + A.chunks[1:]
    store = zarr.open(dst, mode="a", shape=A.shape, chunks=zchunks,
                      dtype=A.dtype)
    ochunks = da.core.normalize_chunks(zchunks, A.shape)
    slices = da.core.slices_from_chunks(ochunks)
    store[slices[i]] = A[slices[i]]
