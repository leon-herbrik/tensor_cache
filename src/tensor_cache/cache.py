from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import zarr
import time
import hashlib


class TensorCache:
    """A cache for storing and retrieving numpy arrays using zarr and fsspec.

    The cache uses a hash-based directory sharding approach to avoid creating
    folders with too many files. Each cached tensor is stored in a path determined
    by the first 4 hex characters of the hash of its id.

    Args:
        base_path: The base path for the cache. Can be a local path (e.g., '/path/to/cache')
            or a remote path using fsspec protocols (e.g., 's3://bucket/cache').
        storage_options: Optional dict of storage options to pass to fsspec for remote paths.

    Example:
        >>> cache = TensorCache('/tmp/my_cache')
        >>> arr = np.array([1, 2, 3])
        >>> cache.set('sample_1', arr)
        >>> retrieved = cache.get('sample_1')
        >>> assert np.array_equal(arr, retrieved)
    """

    def __init__(
        self, base_path: str, storage_options: Optional[Dict[str, Any]] = None
    ) -> None:
        self.base_path = base_path.rstrip("/")
        self.storage_options = storage_options
        self._is_remote = self._check_is_remote(base_path)

    @staticmethod
    def _check_is_remote(path: str) -> bool:
        """Check if path is a remote fsspec URI."""
        return "://" in path and not path.startswith("file://")

    def _get_shard_path(self, item_id: str) -> str:
        """Compute the sharded path for a given item id.

        Uses the first 4 hex characters of a stable hash to create a two-level
        directory structure (e.g., ab/cd/).

        Args:
            item_id: The identifier for the cached item.

        Returns:
            The full path to the zarr array for this item.
        """
        hash_bytes = hashlib.sha256(item_id.encode()).digest()
        hex_hash = hash_bytes.hex()

        shard_1 = hex_hash[:2]
        shard_2 = hex_hash[2:4]

        return f"{self.base_path}/{shard_1}/{shard_2}/{item_id}.zarr"

    def set(
        self, item_id: str, array: np.ndarray, return_stats: bool = False
    ) -> Union[None, Tuple[None, Dict[str, Any]]]:
        """Store a numpy array in the cache.

        Args:
            item_id: The identifier for this cached item.
            array: The numpy array to cache.
            return_stats: If True, return performance statistics along with the result.

        Returns:
            None if return_stats is False, otherwise a tuple of (None, stats_dict) where
            stats_dict contains performance metrics including:
                - duration_seconds: Time taken to store the array
                - array_shape: Shape of the stored array
                - array_dtype: Data type of the stored array
                - array_size_bytes: Size of the array in bytes
        """
        start_time = time.perf_counter()

        zarr_path = self._get_shard_path(item_id)

        # Prepare kwargs for zarr.create_array
        create_kwargs = {
            "store": zarr_path,
            "shape": array.shape,
            "dtype": array.dtype,
            "overwrite": True,
        }

        # Only add storage_options for remote paths
        if self._is_remote and self.storage_options is not None:
            create_kwargs["storage_options"] = self.storage_options

        z = zarr.create_array(**create_kwargs)
        z[:] = array

        duration = time.perf_counter() - start_time

        if return_stats:
            stats = {
                "duration_seconds": duration,
                "array_shape": array.shape,
                "array_dtype": str(array.dtype),
                "array_size_bytes": array.nbytes,
            }
            return None, stats
        return None

    def get(
        self, item_id: str, return_stats: bool = False
    ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Dict[str, Any]]]:
        """Retrieve a numpy array from the cache.

        Args:
            item_id: The identifier for the cached item.
            return_stats: If True, return performance statistics along with the result.

        Returns:
            The cached numpy array (or None if not found) if return_stats is False.
            If return_stats is True, returns a tuple of (array, stats_dict) where stats_dict
            contains performance metrics including:
                - duration_seconds: Time taken to retrieve the array
                - cache_hit: Whether the item was found in cache
                - array_shape: Shape of the retrieved array (if found)
                - array_dtype: Data type of the retrieved array (if found)
                - array_size_bytes: Size of the array in bytes (if found)
        """
        start_time = time.perf_counter()

        zarr_path = self._get_shard_path(item_id)

        result = None
        cache_hit = False

        try:
            # Prepare kwargs for zarr.open_array
            open_kwargs = {"store": zarr_path, "mode": "r"}

            # Only add storage_options for remote paths
            if self._is_remote and self.storage_options is not None:
                open_kwargs["storage_options"] = self.storage_options

            z = zarr.open_array(**open_kwargs)
            result = z[:]
            cache_hit = True
        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            zarr.errors.ArrayNotFoundError,
        ):
            pass

        duration = time.perf_counter() - start_time

        if return_stats:
            stats = {
                "duration_seconds": duration,
                "cache_hit": cache_hit,
            }
            if result is not None:
                stats["array_shape"] = result.shape
                stats["array_dtype"] = str(result.dtype)
                stats["array_size_bytes"] = result.nbytes
            return result, stats
        return result

    def exists(self, item_id: str) -> bool:
        """Check if an item exists in the cache.

        Args:
            item_id: The identifier for the cached item.

        Returns:
            True if the item exists, False otherwise.
        """
        zarr_path = self._get_shard_path(item_id)

        try:
            # Prepare kwargs for zarr.open_array
            open_kwargs = {"store": zarr_path, "mode": "r"}

            # Only add storage_options for remote paths
            if self._is_remote and self.storage_options is not None:
                open_kwargs["storage_options"] = self.storage_options

            zarr.open_array(**open_kwargs)
            return True
        except (FileNotFoundError, zarr.errors.ArrayNotFoundError):
            return False

    def delete(self, item_id: str) -> None:
        """Delete an item from the cache.

        Args:
            item_id: The identifier for the cached item to delete.
        """
        zarr_path = self._get_shard_path(item_id)

        try:
            import shutil

            shutil.rmtree(zarr_path)
        except FileNotFoundError:
            pass
