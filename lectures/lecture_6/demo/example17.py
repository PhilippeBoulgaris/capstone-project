def bisec_search1(L,e):
    if L == []:
        return False
    elif len(L) == 1:
        return L[0] == e
    else:
        half = len(L)//2
        if L[half] > e:
            return bisec_search1(L[:half],e)
        else:
            return bisec_search1(L[half:],e)
        

test_list = [1,4,7,12,67,120,203945]
e = 7

print(bisec_search1(test_list,e))