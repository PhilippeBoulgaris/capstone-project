import random

#################################
## machine abstract data type 
#################################
class machine(object):
    def __init__(self, age):
        self.age = age
        self.name = None
    def get_age(self):
        return self.age
    def get_name(self):
        return self.name
    def set_age(self, newage):
        self.age = newage
    def set_name(self, newname=""):
        self.name = newname
    def __str__(self):
        return "Machine:"+str(self.name)+":"+str(self.age)
        
print("\n---- machine tests ----")
a = machine(4)
print(a)
print(a.get_age())
a.set_name("Audi")
print(a)
a.set_name()
print(a)