from typing import Any, Callable, List
import ray

def run_in_parallel(functions: List[Callable[[], Any]]) -> List[Any]:
    """Runs this list of functions in parallel using ray

    Args:
        functions (List[Callable[[], Any]]): List of functions

    Returns:
        The answers
    """
    ray.init()
    @ray.remote
    def func(i: int):
        return functions[i]()
    ans = ray.get([func.remote(i) for i in range(len(functions))])
    ray.shutdown()
    return ans