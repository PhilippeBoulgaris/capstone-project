def mult_iter(a, b): 
    if b == 1:
        return a
    else:
        return a + mult_iter(a, b-1)

print mult_iter(1,10)