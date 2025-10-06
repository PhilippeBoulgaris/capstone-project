L = [2,1,3,6,3,7,0] # do below in order
print(L)
print("---")

L.remove(2)  # mutates L = [1,3,6,3,7,0]
print(L)
print("---")

L.remove(3)  # mutates L = [1,6,3,7,0]
print(L)
print("---")

del(L[1])    # mutates L = [1,3,7,0]
print(L)
print("---")