import numpy as np



def DisambiguatePose(r_set, c_set, x3D_set):
    """
    https://www.cis.upenn.edu/~cis580/Spring2015/Projects/proj2/proj2.pdf
    To get correct unique camera Pose we need to remove the disambiguity using Cheirality condition.
    The reconstructed points must be in front of cameras.
    """
    best_i = 0
    max_positive_depths = 0

    for i in range(len(r_set)):
        R, C = r_set[i], c_set[i]
        r3 = R[2, :].reshape(1,-1) #3rd column of R
        x3D = x3D_set[i]
        x3D = x3D / x3D[:,3].reshape(-1,1)
        x3D = x3D[:, 0:3]

        #Here we count MAximum Positive Depths
        n_positive_depths = DepthPositivityConstraint(x3D, r3,C)
        # print(n_positive_depths)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths

    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]

    return R, C, x3D 

def DepthPositivityConstraint(x3D, r3, C):
    # r3(X-C) alone doesnt solve the check positivity. z = X[2] must also be +ve 
    n_positive_depths=  0
    for X in x3D:
        X = X.reshape(-1,1) 
        C = C.reshape(-1,1)
        if r3.dot(X-C).T>0 and X[2]>0: 
            n_positive_depths+=1
    return n_positive_depths