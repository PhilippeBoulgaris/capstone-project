x = [0,1,2]
y = x
y[2] = 666
print(x) # [0, 1, 666]
print(y) # [0, 1, 666]

import copy
x = [0,1,2]
y = copy.copy(x)
y[2] = 666
print(x) # [0, 1, 2]
print(y) # [0, 1, 666]

x = [0,1,[2,3]]
y = copy.copy(x)
y[2][0] = 666
print(x) # [0, 1, [666, 3]]
print(y) # [0, 1, [666, 3]]

x = [0,1,[2,3]]
y = copy.deepcopy(x)
y[2][0] = 666
print(x) # [0, 1, [2  , 3]]
print(y) # [0, 1, [666, 3]]