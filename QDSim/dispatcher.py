import multiprocessing

from mock import patch

__all__ = ["repeat_execution", "star_execution"]

def err(exp):
    print(exp)

def repeat_execution(n, f, args=[], kwargs={}):
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    results = manager.list()
    def pool_callback(res):
        results.append(res)
    [pool.apply_async(f, args, kwargs, callback=pool_callback) for i in range(n)] 
    pool.close()
    pool.join()
    return results

def star_execution(f, arg_list, kwarg_list):
    pool = multiprocessing.Pool(4)
    call_with = zip(arg_list,kwarg_list)
    results = []
    for args, kwargs in call_with:
        results.append(pool.apply_async(f, args, kwargs))
    pool.close()
    pool.join()
    return [res.get() for res in results]