from typing import List

from .openml import is_openml_benchmark, load_oml_benchmark
from .file import load_file_benchmark


def benchmark_load(name, benchmark_definition_dirs: List[str]):
    """ Loads the benchmark definition for the 'benchmark' cli input string.

    :param name: the value for 'benchmark'
    :param benchmark_definition_dirs: directories in which benchmark definitions can be found
    :return: a tuple with constraint defaults, tasks, the benchmark path (if it is a local file) and benchmark name
    """
    # Identify where the resource is located, all name structures are clearly defined,
    # but local file benchmark can require probing from disk to see if it is valid,
    # which is why it is tried last.
    if is_openml_benchmark(name):
        benchmark_name, benchmark_path, tasks = load_oml_benchmark(name)
    # elif is_kaggle_benchmark(name):
    else:
        benchmark_name, benchmark_path, tasks = load_file_benchmark(name, benchmark_definition_dirs)

    hard_defaults = next((task for task in tasks if task.name == '__defaults__'), None)
    tasks = [task for task in tasks if task is not hard_defaults]
    return hard_defaults, tasks, benchmark_path, benchmark_name
