import multiprocessing

def error_callback(exp):
    print(exp)

def repeat_execution(f,args,n):
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    results = manager.list()
    def pool_callback(res):
        results.append(res)
    pool.apply_async(f, args, callback=pool_callback,error_callback=error_callback)
    for i in range(n):
        pool.apply_async(f, args, callback=pool_callback)
    pool.close()
    pool.join()
    return results