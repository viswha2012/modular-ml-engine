import functools
import logging
import time


def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.root.handlers and not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(console)

    return logger


logger = get_logger("ml_engine")


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"⏱  {func.__name__}() completed in {elapsed:.4f}s")
        logger.info(f"⏱  {func.__name__}() completed in {elapsed:.4f}s")
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(
                        f"⚠  {func.__name__}() attempt {attempt}/{max_attempts} failed: {e}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            if last_exception is not None:
                raise last_exception

        return wrapper

    return decorator


def log_call(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"→ Calling {func.__name__}()")
            result = func(*args, **kwargs)
            logger.info(f"✓ {func.__name__}() returned successfully")
            return result

        return wrapper

    return decorator
