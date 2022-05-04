"""
1. Polytope NEED NOT be bounded!
2. Non Degenerate
3. Rank(A) = n
4. Given some initial feasible point x
"""

import numpy as np
from sympy import * 

def find_eta_to_vertex(A, b, mask_untight, x, dirc):
    U = A[mask_untight, :]
    temp1 = np.dot(U,x)
    diff_temp = b[mask_untight] - temp1
    temp2 = np.dot(U, dirc)
    temp2[temp2 == 0] = 10**(-8)
    etas = diff_temp/temp2
    etas_neg = etas[etas<0]
    etas_pos = etas[etas>0]
    eta1 = 9999
    eta2 = 9999
    if(etas_neg.size>0):
        eta1 = np.max(etas_neg)
    if(etas_pos.size>0):
        eta2 = np.min(etas_pos)
    if(abs(eta1)<abs(eta2)):
        return eta1
    else:
        return eta2

def find_eta_to_optimum(A, b, mask_untight, x, dirc):
    U = A[mask_untight, :]
    temp1 = np.dot(U,x)
    diff_temp = b[mask_untight] - temp1
    temp2 = np.dot(U, dirc)
    temp2[temp2 == 0] = 10**(-8)
    etas = diff_temp/temp2
    mask_positive = etas > 0
    etas = etas[mask_positive]
    if(etas.size == 0):
        return -1
    return np.min(etas)


def rows_indices(A, b, x):
    #Let's find Tight Rows
    temp_vector = np.dot(A, x)
    #Indices of tight rows
    mask_tight = (temp_vector == b).reshape(b.shape[0])
    #Let's find untight rows indices as well
    mask_untight = (temp_vector < b).reshape(b.shape[0])
    return mask_tight, mask_untight

def direction_from_vertex_to_vertex(A, b, c, x, m, n, alphas):
    #Let's find Tight Rows
    mask_tight, mask_untight = rows_indices(A, b, x)
    T = A[mask_tight, :]
    
    #Find inverse of T
    T_inv = np.linalg.inv(T)
    
    #Find index of first negative alpha (without loss of generality) for which c.(-column) > 0
    temp_check = np.dot((-1 * T_inv.T), c)
    mask_alphas = temp_check > 0
    mask_alphas = mask_alphas.reshape(mask_alphas.shape[0])
    candidate_cols = T_inv.T[mask_alphas, :]
    candidate_dir = candidate_cols[0,:].T
#     neg_index = (np.where(alphas < 0))[0][0]
    
    #The direction is opposite to that indexed column of T_inv
    dirc = -1 * candidate_dir
    dirc = dirc.reshape(dirc.shape[0],1)
    
    return dirc/np.linalg.norm(dirc), mask_tight, mask_untight
    
    

def direction_from_point_to_vertex(A, b, c, x, m, n):
    #Let's find Tight Rows
    mask_tight, mask_untight = rows_indices(A, b, x)
    T = A[mask_tight, :]

    #Find vector "y" such that T.y = 0
    null_space = np.array(Matrix(T).nullspace())
    assert null_space.size > 0, \
    "No Solution Found!"
    coeffs = np.random.randint(low = 1, high = 5 , size = null_space.shape[0]).reshape(null_space.shape[0], 1, 1)
    y = np.array(np.sum(null_space * coeffs, axis = 0)).astype(float)
    return y/np.linalg.norm(y), mask_tight, mask_untight
    

def isOptimumVertex(A, b, c, x, m, n):
    #Let's find Tight Rows
    mask_tight, mask_untight = rows_indices(A, b, x)
    T = A[mask_tight, :]
    
    #Assuming x is a vertex, check if c is non-negative linear combination of rows of T
    alphas = np.dot(np.linalg.inv(T.T), c)
    n_neg_alphas = np.sum(alphas<0)
    
    if(n_neg_alphas == 0):
        return alphas, True, mask_tight, mask_untight
    else:
        return alphas, False, mask_tight, mask_untight
    
def isVertex(A, b, c, x, m, n):
    #Let's find Tight Rows
    mask_tight, mask_untight = rows_indices(A, b, x)
    T = A[mask_tight, :]
    
    rank = 0
    if(T.size != 0):
        rank = np.linalg.matrix_rank(T)
        
    #Check if rank(T) == n
    if(rank == n):
        return True, mask_tight, mask_untight
    else:
        return False, mask_tight, mask_untight
    

def simplex_algorithm(A, b, c, x, m, n):
    #eta is step size to move in direction from non-vertex to vertex
    #t is step size to move in direction from vertex to optimum vertex
    while(True):
        #Step 1
        print("x = {}".format(x))
        is_Vertex, mask_tight, mask_untight = isVertex(A, b, c, x, m, n)
        if(is_Vertex):
            alphas, isOpt, mask_tight, mask_untight = isOptimumVertex(A, b, c, x, m, n)
            #Step 2
            if(isOpt):
                print("Optimum vertex is x = {}".format(x))
                print("Maximum value of c.x = {}".format(np.dot(c.T, x).item()))
                return
            else:
                dir_to_opt, mask_tight, mask_untight = direction_from_vertex_to_vertex(A, b, c, x, m, n, alphas)
                eta = find_eta_to_optimum(A, b, mask_untight, x, dir_to_opt)
                if(eta == -1):
                    print("Cost is Unbounded!")
                    return
                x = x + (eta * dir_to_opt)
        else:
            dir_to_vertex, mask_tight, mask_untight = direction_from_point_to_vertex(A, b, c, x, m, n)
            eta = find_eta_to_vertex(A, b, mask_untight, x, dir_to_vertex)
            x = x + (eta * dir_to_vertex)
            
            
            
m = int(input("m = "))
n = int(input("n = "))

A = np.zeros((m, n))
b = np.zeros((m,1))
c = np.zeros((n,1))
x = np.zeros((n,1))

#Input A
print("Input Matrix A: ")
for i in range(0,m): 
    temp = input().split()
    assert len(temp) == n, \
    "Enter correct number of elements in column!"
    A[i,:] = np.array([float(j) for j in temp]) 

#Input b
print("Input Vector b: ")
for i in range(0,m): 
    b[i][0] = float(input())
    
#Input c
print("Input Cost Vector c: ")
for i in range(0,n): 
    c[i][0] = float(input())
    
#Initial Feasible Point
print("Input Initial Feasible Point x: ")
for i in range(0,n): 
    x[i][0] = float(input())
    

simplex_algorithm(A, b, c, x, m, n)
