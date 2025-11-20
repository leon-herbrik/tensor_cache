# tensor-cache

Simple tensor caching library using zarr 3.x and fsspec for local and remote storage.

## Installation

```bash
pip install zarr>=3.0.0 numpy>=1.20.0 fsspec>=2021.0.0

# For S3
pip install s3fs

# For Google Cloud Storage
pip install gcsfs
```

## Usage

```python
from tensor_cache import TensorCache
import numpy as np

# Local storage
cache = TensorCache("/tmp/my_cache")

# Remote storage (S3, GCS, Azure, etc.)
cache = TensorCache(
    "s3://my-bucket/cache",
    storage_options={"anon": True}  # or credentials
)

# Store and retrieve
arr = np.random.rand(100, 100)
cache.set("sample_1", arr)
retrieved = cache.get("sample_1")

# Check existence
if cache.exists("sample_1"):
    cache.delete("sample_1")

# Get performance stats
_, stats = cache.set("sample_2", arr, return_stats=True)
print(f"Stored in {stats['duration_seconds']:.4f}s")
```

## Features

- Hash-based sharding to avoid too many files per directory
- Support for local and remote storage (S3, GCS, Azure via fsspec)
- Optional performance statistics
- Compatible with zarr 3.x

## Storage Options

```python
# S3 with credentials
TensorCache("s3://bucket/cache", storage_options={
    "key": "ACCESS_KEY",
    "secret": "SECRET_KEY"
})

# Google Cloud Storage
TensorCache("gs://bucket/cache", storage_options={
    "token": "/path/to/credentials.json"
})

# Azure
TensorCache("az://container/cache", storage_options={
    "account_name": "name",
    "account_key": "key"
})
```