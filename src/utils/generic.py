import time

def prints(content, status):
    if status is 0:
        msg = '.'
    elif status is 1:
        msg = 'DONE'
    elif status is 2:
        msg = 'WARN'
    elif status is 3:
        msg = 'FAIL'
    print(f'\n[{msg}]\t{content}\n')

def timer(func):
    """
    Record execution time of any function with timer decorator
    Usage: just decorate a function when building it, the 
    decorator will be called every time the function is executed.
    # build the function
    @timer
    def some_function(some_arg):
        # do_something
        return 'foo'
        
    # call it
    some_function('boo')
    # output:
    >> Function 'some_function' finished after 0.01 seconds.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        duration = time.time() - start
        print("Function '{}' finished after {:.4f} seconds."\
              .format(func.__name__, duration))
        return results
    return wrapper