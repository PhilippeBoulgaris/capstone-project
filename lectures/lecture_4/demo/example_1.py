def fib(n):
    """Returns the nth Fibonacci number."""
    a = 0
    b = 1
    for i in range(n):
      tmp = a
      a = a + b
      b = tmp
    return a




print("function call from the original Function -- fib(7) =",fib(7))

print(fib(10))
print(fib(20))
