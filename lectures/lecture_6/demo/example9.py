import time

def c_to_f(c):
    return c*9.0/5.0 + 32.0

t0 = time.perf_counter()

size = 100000
for i in range(size):
    c = int(100/size)*i
    c_to_f(c)


t1 = time.perf_counter()- t0

print("runtime: ", t1, "[sec]")
