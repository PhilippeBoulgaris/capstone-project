def bisec_search2(L,e):
    def bisec_search_helper(L, e, low, high):
        if high == low:
            return L[low] == e
        mid = (low + high ) //2
        if L[mid] == e:
            return True
        elif L[mid] > e:
            if low == mid: ##nothing left to bisec_search
                return False
            else:
                return bisec_search_helper(L, e, low, mid -1 )
        else:
            return bisec_search_helper(L, e, mid + 1, high)
    if len(L) == 0:
        return False
    else:
        return bisec_search_helper(L,e,0,len(L) - 1 )
    
        

test_list = [1,4,7,12,67,120,203945]
e = 7

print(bisec_search2(test_list,e))