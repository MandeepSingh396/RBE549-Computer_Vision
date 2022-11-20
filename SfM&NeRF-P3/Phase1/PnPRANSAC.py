import numpy as np
from LinearPnP import PnP

def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def PnPError(feature, X, R, C, K):
    u,v = feature
    pts = X.reshape(1,-1)
    X = np.hstack((pts, np.ones((pts.shape[0],1))))
    X = X.reshape(4,1)
    C = C.reshape(-1,1)
    P = ProjectionMatrix(R,C,K)
    p1, p2, p3 = P
    p1, p2, p3 = p1.reshape(1,4), p2.reshape(1,4), p3.reshape(1,4)
    u_proj = np.divide(p1.dot(X), p3.dot(X))
    v_proj = np.divide(p2.dot(X), p3.dot(X))

    x_proj = np.hstack((u_proj, v_proj))
    x = np.hstack((u,v))
    err = np.linalg.norm(x-x_proj)

    return err

def PnPRANSAC(K,features,x3D, iter=1000, thresh=5):
    inliers_thresh = 0
    R_best, t_best = None, None
    n_rows = x3D.shape[0]

    for i in range(0, iter):

        #We rand select 6 pts
        rand_indices = np.random.choice(n_rows, size=6)
        X_set, x_set = x3D[rand_indices], features[rand_indices]

        #WE get R and C from PnP function
        R, C = PnP(X_set, x_set, K)

        indices = []
        if R is not None:
            for j in range(n_rows):
                feature = features[j]
                X = x3D[j]
                error = PnPError(feature, X, R, C, K)

                if error < thresh:
                    indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            R_best = R
            t_best = C

    return R_best, t_best
                