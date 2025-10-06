# empty tuple
te = ()

t = (2,"HEC",3) 
print(t[0])            #evaluates to 2

a=(2,"HEC",3) + (5,6) #  evaluates to (2,"HEC",3,5,6)
print(a)

b = t[1:2]       #  slice tuple, evaluates to ("HEC",)  Note: the extra comma means a tuple with 1 element
print(b)

c = t[1:3]       #slice tuple, evaluates to ("HEC",3)
print(c)

print(len(t))	     #evaluates to 3

t[1] = 4     #gives an error, cannot modify object