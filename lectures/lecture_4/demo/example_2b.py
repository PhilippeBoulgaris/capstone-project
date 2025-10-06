def f(y):
    x = 1
    x += 1
    print(x)
x = 5
f(x)
print(x)
print("----")

def g(y):
    print(x)
    print(x+1)
x = 5
g(x)
print(x)
print("----")


def h(y):
    pass
    #x += 1 #leads to an error without line `global x` inside h
x = 5
h(x)
print(x)
print("----")
