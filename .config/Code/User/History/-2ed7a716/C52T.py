import numpy as np

# Given a list “l” of numbers and a scalar number “s”, develop an efficient algorithm which returns all pairs of numbers in the list “l” that sum up to the scalar number “s”.

s = 6
l = [1, 2, 4, 5, 3, 3]


#r = [0 for i in range(len(l)**2)]
r = []

idx = 0
for i_idx in range(len(l)):
    for j_idx in range(i_idx+1, len(l)):
        if l[i_idx]+l[j_idx] == s:
            
            #for pair in r:

                
            r.append((i_idx, j_idx))
            

print(r)
            


            