import numpy as np

def getEssentialMatrix(K,F):
    E = K.T.dot(F).dot(K)
    U,S,V = np.linalg.svd(E)
    S = [1,1,0]

    return np.dot(U,np.dot(np.diag(S),V))