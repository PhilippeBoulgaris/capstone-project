L1 = [2,1,3]
print(L1)
print("-----")

L2 = [4,5,6]
L3 = L1 + L2         # L3 is [2,1,3,4,5,6], L1, L2 unchanged
print(L1)
print("-----")

L1.extend([0,6])     # mutated L1 to [2,1,3,0,6] 
print(L1)
print("-----")