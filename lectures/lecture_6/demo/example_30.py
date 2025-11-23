try:
    age = int(input("Please enter your age: "))
except ValueError as err:
    print(err)