"""
Cache manager for avoiding redundant processing in data extraction pipeline.
"""
import os
import json
import hashlib
import pickle
import time
from typing import Any, Dict, Optional
from functools import wraps

class CacheManager:
    """Manages caching for extraction results to avoid redundant processing."""
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache configuration
        self.max_age = 24 * 60 * 60  # 24 hours
        self.max_size = 1000  # Maximum number of cached items
        
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.cache")
    
    def _is_valid(self, cache_path: str) -> bool:
        """Check if a cache file is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        # Check age
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age > self.max_age:
            os.remove(cache_path)
            return False
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_path = self._get_cache_path(key)
        
        if not self._is_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # Remove corrupted cache
            try:
                os.remove(cache_path)
            except:
                pass
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def clear(self) -> None:
        """Clear all cache."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                except:
                    pass
    
    def cleanup(self) -> None:
        """Remove old cache files."""
        current_time = time.time()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    if current_time - os.path.getmtime(filepath) > self.max_age:
                        os.remove(filepath)
                except:
                    pass

# Global cache instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def cache_result(key_func=None, ttl: int = None):
    """
    Decorator for caching function results.
    
    Args:
        key_func: Function to generate cache key from arguments
        ttl: Time to live in seconds (overrides default)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cache = get_cache_manager()
            result = cache.get(cache_key)
            
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator

# Specific cache key generators
def pdf_cache_key(filepath: str, page_num: int) -> str:
    """Generate cache key for PDF page processing."""
    # Include file modification time and size for invalidation
    try:
        mtime = os.path.getmtime(filepath)
        size = os.path.getsize(filepath)
        return f"pdf_{filepath}_{page_num}_{mtime}_{size}"
    except:
        return f"pdf_{filepath}_{page_num}"

def table_cache_key(table_data: tuple) -> str:
    """Generate cache key for table extraction."""
    # Hash the table data for consistency
    table_str = json.dumps(table_data, sort_keys=True)
    return f"table_{hashlib.md5(table_str.encode()).hexdigest()}"

def plot_cache_key(img_shape: tuple, axes_info: dict) -> str:
    """Generate cache key for plot digitization."""
    key_data = {
        'shape': img_shape,
        'axes': axes_info
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return f"plot_{hashlib.md5(key_str.encode()).hexdigest()}"
