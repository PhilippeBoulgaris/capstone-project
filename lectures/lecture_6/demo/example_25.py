#######################################
## EXAMPLE: Buggy code to reverse a list
#######################################
def rev_list_buggy(L):
   """
   input: L, a list
   Modifies L such that its elements are in reverse order
   returns: nothing
   """
   for i in range(len(L)):
       j = len(L) - i
       L[i] = temp
       L[i] = L[j]
       L[j] = L[i]

# FIXES: --------------------------
# temp unknown
# list index out of range -> sub 1 to j
# get same list back -> iterate only over half
# --------------------------


### debugged code
def rev_list(L):
    """
    input: L, a list
    Modifies L such that its elements are in reverse order
    returns: nothing
    """
    for i in range(len(L)//2):
        j = len(L) - i - 1
        temp = L[i]
        L[i] = L[j]
        L[j] = temp



L = [1,2,3,4]
rev_list_buggy(L)  #call buggy code
#rev_list(L)       #call correct code
print(L)

