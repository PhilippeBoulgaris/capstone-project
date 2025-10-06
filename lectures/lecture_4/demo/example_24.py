my_dict = {}   #empty dictionary

grades = {'Tom':6.0, 'Keith':4.5, 'Marry':5.2, 'Megan':4.9}

print (grades['Marry'])

grades['Mickey'] = 5.0  #add an entry 
print (grades)
print(grades['Mickey'])

test = 'Tom' in grades
print(test)

del(grades['Marry'])    #remove entries
print (grades)

a = grades.keys()       #get all keys
print(a)

b = grades.values()         #get all values
print (b)

d = {4:{1:0}, (1,3):"twelve", 'const':[3.14,2.7,8.44]}	 #a heterogenous dict
print(d.keys())
print(d[4])
print(d.values())