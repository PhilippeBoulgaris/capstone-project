def quotient_and_remainder(x, y):
    q = x // y
    r = x % y
    return (q, r)
    
(quot, rem) = quotient_and_remainder(5,3)
print(quot)
print(rem)