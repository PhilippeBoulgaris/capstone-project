def genSubsets(L):
    res = []
    if len(L) == 0:
        return [[]] ##list of empty sets
    smaller = genSubsets(L[:-1]) #all subsets without last element
    extra = L[-1:]
    new =  []
    for small in smaller:
        new.append(small + extra) #for all smaller sol, add one with last el.
    return smaller + new


Ltest = [1,2,3,4,5]
print(genSubsets(Ltest))
