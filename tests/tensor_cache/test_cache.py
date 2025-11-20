import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from tensor_cache import TensorCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cache(temp_cache_dir):
    """Create a TensorCache instance with a temporary directory."""
    return TensorCache(temp_cache_dir)


def test_set_and_get_simple_array(cache):
    """Test storing and retrieving a simple 1D array."""
    arr = np.array([1, 2, 3, 4, 5])
    cache.set("test_1", arr)
    
    retrieved = cache.get("test_1")
    assert retrieved is not None
    assert np.array_equal(arr, retrieved)


def test_set_and_get_multidimensional_array(cache):
    """Test storing and retrieving multidimensional arrays."""
    arr_2d = np.random.rand(10, 20)
    cache.set("test_2d", arr_2d)
    
    retrieved = cache.get("test_2d")
    assert retrieved is not None
    assert np.array_equal(arr_2d, retrieved)
    
    arr_3d = np.random.rand(5, 10, 15)
    cache.set("test_3d", arr_3d)
    
    retrieved = cache.get("test_3d")
    assert retrieved is not None
    assert np.array_equal(arr_3d, retrieved)


def test_get_nonexistent_item(cache):
    """Test that getting a non-existent item returns None."""
    result = cache.get("nonexistent")
    assert result is None


def test_different_dtypes(cache):
    """Test caching arrays with different data types."""
    dtypes = [np.int32, np.float32, np.float64, np.bool_, np.uint8]
    
    for dtype in dtypes:
        arr = np.array([1, 2, 3], dtype=dtype)
        cache.set(f"test_{dtype.__name__}", arr)
        
        retrieved = cache.get(f"test_{dtype.__name__}")
        assert retrieved is not None
        assert retrieved.dtype == dtype
        assert np.array_equal(arr, retrieved)


def test_exists(cache):
    """Test the exists method."""
    arr = np.array([1, 2, 3])
    
    assert not cache.exists("test_exists")
    
    cache.set("test_exists", arr)
    assert cache.exists("test_exists")
    
    assert not cache.exists("nonexistent")


def test_delete(cache):
    """Test deleting cached items."""
    arr = np.array([1, 2, 3])
    cache.set("test_delete", arr)
    
    assert cache.exists("test_delete")
    
    cache.delete("test_delete")
    assert not cache.exists("test_delete")
    assert cache.get("test_delete") is None


def test_delete_nonexistent(cache):
    """Test that deleting a non-existent item doesn't raise an error."""
    cache.delete("nonexistent")


def test_overwrite_existing(cache):
    """Test overwriting an existing cached item."""
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6, 7])
    
    cache.set("test_overwrite", arr1)
    retrieved1 = cache.get("test_overwrite")
    assert np.array_equal(arr1, retrieved1)
    
    cache.set("test_overwrite", arr2)
    retrieved2 = cache.get("test_overwrite")
    assert np.array_equal(arr2, retrieved2)


def test_multiple_items(cache):
    """Test storing and retrieving multiple different items."""
    items = {
        "item_1": np.array([1, 2, 3]),
        "item_2": np.random.rand(5, 5),
        "item_3": np.array([[1, 2], [3, 4]]),
        "item_4": np.zeros((10, 10)),
    }
    
    for item_id, arr in items.items():
        cache.set(item_id, arr)
    
    for item_id, arr in items.items():
        retrieved = cache.get(item_id)
        assert retrieved is not None
        assert np.array_equal(arr, retrieved)


def test_shard_path_generation(cache):
    """Test that shard paths are generated correctly."""
    item_id = "test_item"
    shard_path = cache._get_shard_path(item_id)
    
    assert item_id in shard_path
    assert ".zarr" in shard_path
    
    path_parts = shard_path.split('/')
    assert len(path_parts[-3]) == 2
    assert len(path_parts[-2]) == 2


def test_different_ids_different_paths(cache):
    """Test that different IDs generate different shard paths."""
    ids = ["id_1", "id_2", "id_3", "another_id"]
    paths = [cache._get_shard_path(item_id) for item_id in ids]
    
    assert len(set(paths)) == len(paths)


def test_empty_array(cache):
    """Test caching empty arrays."""
    arr = np.array([])
    cache.set("empty", arr)
    
    retrieved = cache.get("empty")
    assert retrieved is not None
    assert len(retrieved) == 0


def test_large_array(cache):
    """Test caching a large array."""
    arr = np.random.rand(1000, 1000)
    cache.set("large", arr)
    
    retrieved = cache.get("large")
    assert retrieved is not None
    assert np.array_equal(arr, retrieved)


def test_special_characters_in_id(cache):
    """Test that IDs with special characters work correctly."""
    arr = np.array([1, 2, 3])
    special_ids = ["id-with-dash", "id_with_underscore", "id.with.dots"]
    
    for item_id in special_ids:
        cache.set(item_id, arr)
        retrieved = cache.get(item_id)
        assert retrieved is not None
        assert np.array_equal(arr, retrieved)


def test_set_with_stats(cache):
    """Test set method with return_stats=True."""
    arr = np.array([1, 2, 3, 4, 5])
    result, stats = cache.set("test_stats", arr, return_stats=True)
    
    assert result is None
    assert 'duration_seconds' in stats
    assert stats['duration_seconds'] >= 0
    assert stats['array_shape'] == arr.shape
    assert stats['array_dtype'] == str(arr.dtype)
    assert stats['array_size_bytes'] == arr.nbytes


def test_get_with_stats_hit(cache):
    """Test get method with return_stats=True for cache hit."""
    arr = np.random.rand(10, 10)
    cache.set("test_get_stats", arr)
    
    result, stats = cache.get("test_get_stats", return_stats=True)
    
    assert result is not None
    assert np.array_equal(arr, result)
    assert 'duration_seconds' in stats
    assert stats['duration_seconds'] >= 0
    assert stats['cache_hit'] is True
    assert stats['array_shape'] == arr.shape
    assert stats['array_dtype'] == str(arr.dtype)
    assert stats['array_size_bytes'] == arr.nbytes


def test_get_with_stats_miss(cache):
    """Test get method with return_stats=True for cache miss."""
    result, stats = cache.get("nonexistent", return_stats=True)
    
    assert result is None
    assert 'duration_seconds' in stats
    assert stats['duration_seconds'] >= 0
    assert stats['cache_hit'] is False
    assert 'array_shape' not in stats
    assert 'array_dtype' not in stats
    assert 'array_size_bytes' not in stats


def test_stats_timing_makes_sense(cache):
    """Test that timing statistics are reasonable."""
    small_arr = np.array([1, 2, 3])
    large_arr = np.random.rand(1000, 1000)
    
    _, small_stats = cache.set("small", small_arr, return_stats=True)
    _, large_stats = cache.set("large", large_arr, return_stats=True)
    
    assert small_stats['duration_seconds'] < 10
    assert large_stats['duration_seconds'] < 10
    
    _, get_stats = cache.get("small", return_stats=True)
    assert get_stats['duration_seconds'] < 10
