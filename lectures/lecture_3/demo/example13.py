cube = 27.8
epsilon = 0.1
guess = 0.0
increment = 0.01
num_guesses = 0
# look for close enough answer and make sure
# didn't accidentally skip the close enough bound and thus overshoot
while abs(guess**3 - cube) >= epsilon and guess <= cube:
   guess += increment
   num_guesses += 1
print('num_guesses =', num_guesses)
if abs(guess**3 - cube) >= epsilon:
   print('Failed on cube root of', cube, "with these parameters.")
else:
   print(guess, 'is close to the cube root of', cube)