import multiprocessing

def error_callback(exp):
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
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    results = manager.list()
    call_with = zip(arg_list,kwarg_list)
    def pool_callback(res):
        results.append(res)
    for args,kwargs in call_with:
        print(args)
        print(kwargs)
        pool.apply_async(f, args=args, kwds=kwargs, callback=pool_callback,
                              error_callback=error_callback)

    pool.close()
    pool.join()
    return results