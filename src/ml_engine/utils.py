import functools
import logging
import time
from pathlib import Path


def get_logger(name: str, log_file: str = "logs/main.log") -> logging.Logger:

    log_path = Path(__file__).resolve().parent.parent.parent / log_file
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"⏱  {func.__name__}() completed in {elapsed:.4f}s")
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
