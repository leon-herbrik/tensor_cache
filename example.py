"""Example usage of tensor-cache library."""

from tensor_cache import TensorCache
import numpy as np


def main():
    cache = TensorCache("/tmp/my_tensor_cache")
    
    sample_tensor = np.random.rand(100, 100)
    
    _, set_stats = cache.set("sample_1", sample_tensor, return_stats=True)
    print(f"Stored tensor with shape: {sample_tensor.shape}")
    print(f"Store operation took: {set_stats['duration_seconds']:.6f} seconds")
    print(f"Array size: {set_stats['array_size_bytes']} bytes")
    
    retrieved, get_stats = cache.get("sample_1", return_stats=True)
    if retrieved is not None:
        print(f"\nRetrieved tensor with shape: {retrieved.shape}")
        print(f"Retrieval took: {get_stats['duration_seconds']:.6f} seconds")
        print(f"Cache hit: {get_stats['cache_hit']}")
        print(f"Arrays are equal: {np.array_equal(sample_tensor, retrieved)}")
    
    _, miss_stats = cache.get("nonexistent", return_stats=True)
    print(f"\nCache miss took: {miss_stats['duration_seconds']:.6f} seconds")
    print(f"Cache hit: {miss_stats['cache_hit']}")
    
    if cache.exists("sample_1"):
        print("\nCache entry exists!")
    
    cache.delete("sample_1")
    print(f"After deletion, exists: {cache.exists('sample_1')}")
    
    cache_s3 = TensorCache("s3://my-bucket/cache")
    print("\nCreated cache with S3 backend (requires s3fs)")


if __name__ == "__main__":
    main()
