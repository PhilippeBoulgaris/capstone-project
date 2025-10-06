x = 12/3 - 2      # this is a comment
y = "Hola"
z = 3.14          # another comment

if (y == "Hola" or z >= 3):
    x = x + 2
    y = y + " mundo!" # string concatenation    
    print(y)
    print(x)
    
year, month , day  = 1943, 6, 15
hour, minute, second = 23, 6, 54
if 1900 < year < 2100 and 1 <= month <= 12  \
   and 1 <= day <= 31 and 0 <= hour < 24    \
   and 0 <= minute < 60 and 0 <= second < 60:
      print("Looks like a valid date!")
