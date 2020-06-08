import numpy as np

np.random.seed(0)

X = np.loadtxt("X.txt")
Y = np.loadtxt("Y.txt")

np.save('X', X)
np.save('Y', Y)


splits = np.random.permutation(X.shape[ 0 ])

for i in range(99):
    splits = np.vstack((splits, np.random.permutation(X.shape[ 0 ])))

np.save('splits', splits)

noise01 = np.random.normal(size = ((100, X.shape[ 0 ], X.shape[ 1 ]))) * np.sqrt(0.1)
np.save('noise01', noise01)
noise25 = np.random.normal(size = ((100, X.shape[ 0 ], X.shape[ 1 ]))) * np.sqrt(0.25)
np.save('noise25', noise25)
noise5 = np.random.normal(size = ((100, X.shape[ 0 ], X.shape[ 1 ]))) * np.sqrt(0.5)
np.save('noise5', noise5)

