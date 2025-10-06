x = dict(a=1, b=2, c='three')
x = {'a': 1, 'b': 2, 'c': 'three'}
print(x)
print("----")

# access via []
x['a'] == 1
print(x)
print("----")

# creating new entries
# any hashable type can be a key
x[1] = 4
print(x)
print("----")

# accessing keys, values or both
# order is not preserved
x.keys() # ['a', 'c', 1, 'b']
print(x.keys())
print("----")

x.values() # [1, 'three', 4, 2]
print(x.values())
print("----")

x.items() # [('a', 1), (c, 'three'), (1, 4), ('b', 2)] 
print(x.items())
print("----")