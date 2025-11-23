def fib_recursive(n):
    """ assumes n an int >=0 """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_recursive(n -1 ) + fib_recursive(n - 2)
    
    
n_iter = 10
print(fib_recursive(n_iter))