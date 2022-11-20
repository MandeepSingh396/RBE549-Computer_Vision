import numpy as np

K = np.array([[531.122155322710, 0 ,407.192550839899],[0, 531.541737503901, 313.308715048366],[0,0,1]])

R = np.identity(3)
I = np.identity(3)
C = np.zeros((3,1)).reshape(3,1)
A = np.array([0,0,1,0]).reshape(-1,1)
B = np.array([ 2.28461958,-0.35034747,4.52122536,1]).reshape(1,4)
P = np.dot(K,np.dot(R,np.hstack((I,-C))))

print(A.dot(B))