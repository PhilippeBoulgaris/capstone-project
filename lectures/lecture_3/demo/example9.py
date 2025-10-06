for item in [0,"a",7,1j]:
    print(item)

for letter in "StRiNg":
    print(letter)

for i in range(5):
    print(i)

lst = ["Suzuki","Kawasaki","Aprilia","Ducati"]
# use enumerate below!!!
# for i in range(len(lst)):
#     print(i,lst[i])
for (i,item) in enumerate(lst):
    print(i,item)
