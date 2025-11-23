def search_for_elmt(L, e):
    for i in L:
        if i == e:
            return True
    return False
    

testList = [1, 3, 4, 5, 9, 18, 27]
e_test = 27

#test if element 

print(search_for_elmt(testList,e_test))