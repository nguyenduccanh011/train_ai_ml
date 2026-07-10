import time
from functools import wraps

from loguru import logger


class MetricsCollector:
    def __init__(self):
        self.metrics = {}

    def record_endpoint(self, endpoint_name):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start
                    logger.info(f"Endpoint {endpoint_name} completed in {duration:.2f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start
                    logger.error(f"Endpoint {endpoint_name} failed after {duration:.2f}s: {e}")
                    raise

            return wrapper

        return decorator


metrics = MetricsCollector()
