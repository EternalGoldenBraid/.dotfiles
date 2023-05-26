import numpy as np

# Given a list “l” of numbers and a scalar number “s”, develop an efficient algorithm which returns all pairs of numbers in the list “l” that sum up to the scalar number “s”.

s = 6
l = [1, 2, 4, 5, 3, 3]


#r = [0 for i in range(len(l)**2)]
r = []

idx = 0
for i in l:
    for j in l:
        if i+j == s:
            r.append(i, j)
            


            