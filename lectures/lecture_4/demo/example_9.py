# list is built-in
x = [0, 1, 2, 3, 3]

x[2] == 2 # access via [] index operator

x.insert(0, 5) # index, value 
# x == [5, 0, 1, 2, 3, 3]
print(x)
print("-----")

# remove by index - returns value
x.pop(0) # returns 5 
# x == [0, 1, 2, 3, 3]
print(x)
print("-----")

x = [0, 1, 2, 'three'] # can contain arbitrary types!

# access from the back with negative indices
x[-2] == 2 
print(x)
print("-----")

# adding lists concatenates them
x += [4, 5, 6] # x == [0, 1, 2, 'three', 4, 5, 6]
print(x)
print("-----")

# slicing [start:end + 1]
x[1:4] == [1, 2, 'three'] 
print(x)
print("-----")

# slicing with a stride [start:end + 1:step]
x[0:7:2] == x[::2] == [0, 2, 4, 6]
print(x)
print("-----")

# reverse slicing
x[-1:0:-2] == [6, 4, 2]
x[::-2] == [6, 4, 2, 0]
print(x)
print("-----")