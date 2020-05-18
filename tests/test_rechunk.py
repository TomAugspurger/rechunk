from pathlib import Path
import pytest
from rechunk import __version__

import dask.array as da
from dask.distributed import Client
import rechunk


@pytest.fixture(scope="module")
def client():
    with Client(n_workers=2, threads_per_worker=1):
        yield client


def test_version():
    assert __version__


def test_basic(client, tmpdir):
    a = da.random.random(size=(4, 8, 8), chunks=(1, 8, 8))
    tmpdir = Path(tmpdir)
    original = str(tmpdir / "original.zarr")
    a.to_zarr(original)

    b = rechunk.rechunk(
        original,
        str(tmpdir / "split.zarr"),
        str(tmpdir / "final.zarr"),
        split_chunks=(4, 4),
    )
    da.utils.assert_eq(a, b)
