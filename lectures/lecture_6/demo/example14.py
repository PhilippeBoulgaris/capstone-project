def search(L, e):
    for i in range(len(L)):
        if L[i] == e:
            return True
        if L[i] > e:
            return False
    return False

testList = [1, 3, 4, 5, 9, 18, 27]
e = 18

print(search(testList,e))
