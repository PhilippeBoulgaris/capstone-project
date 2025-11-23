import math
 
class Circle(object):
 
    def __init__(self, radius):
        self.radius = radius
 
    def __str__(self):
        """ Returns a string representation of self """
        return "< Circle with Radius " + str(self.radius) + ">"  
 
    def get_area(self):
        return math.pi * self.radius ** 2
 
    def get_circumference(self):
        return 2 * math.pi * self.radius

    def get_ratios(self, other):
        area_frac = (self.radius/other.radius)**2.0
        circumference_frac = self.radius/other.radius
        return_tuple = (area_frac,circumference_frac)
        return return_tuple

radius_1 = 1.0
my_circle = Circle(radius_1)
 
print("Circle of radius is",my_circle.radius)
print("Area of circle:", my_circle.get_area())
print("Area of perimeter of circle:",  my_circle.get_circumference())

radius_2 = 2.0
my_circle2 = Circle(radius_2)
print("Circle of radius is",my_circle2.radius)
print("Area of circle:", my_circle2.get_area())

print(" Ratios of the two Areas and Circumferences ", my_circle.get_ratios(my_circle2))
