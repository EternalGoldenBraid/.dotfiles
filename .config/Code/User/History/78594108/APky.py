# Given a list of length N, containing numbers in the range [0, N-1], 
# develop an efficient algorithm which returns all the duplicates in the list.
# Discuss memory and computational complexity


N = 5
L = [0, 0, 1, 4, 4]

r = []

leng = len(L)
for i_idx in range(leng):

    if i_idx == 0: continue

    i = L[i_idx]
    for j_idx in range(i_idx-1, i_idx+1):
        j = L[j_idx]
        if i==j:
            r.append((i,j))
            
        
        
        
        



