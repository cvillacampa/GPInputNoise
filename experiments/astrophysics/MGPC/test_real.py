import numpy as np
import tensorflow as tf
import sys
sys.path.append("../../..")
from gp_input_noise.mgpc import MGPC
from gp_input_noise.rbf_ard import RBF_ARD


np.random.seed(0)

fold = int(sys.argv[1])

data = np.genfromtxt("../data/collected_data_no_NANs_filtered.dat").astype(np.float32)
indices = np.genfromtxt("../data/data_splits.txt")[ fold ].astype(np.int32)

n_train = int(len(indices) * 0.9)
indices_train = indices[:n_train]
indices_test = indices[n_train:]

cols_data = [0, 2, 3, 4, 6, 8 ]

data_train, data_test = data[indices_train], data[indices_test]

X_train, y_train, X_test, y_test = data_train[:, cols_data], data_train[:, -1].astype(np.int32), data_test[:, cols_data], data_test[:, -1].astype(np.int32)
y_train = y_train[:, None]
y_test = y_test[:, None]

mean_train = np.mean(X_train, 0)
std_train = np.std(X_train, 0)

X_train = (X_train - mean_train) / std_train
X_test = (X_test - mean_train) / std_train

noise_var_train = np.zeros_like(X_train) + 1e-5
noise_var_train[:,0] = (data_train[:, 1 ] / np.std(data_train[ :, 0 ]))**2
noise_var_train[:,3] = (data_train[:, 5 ] / np.std(data_train[ :, 4 ]))**2
noise_var_train[:,4] = (data_train[:, 7 ] / np.std(data_train[ :, 6 ]))**2

noise_var_test = np.zeros_like(X_test) + 1e-5
noise_var_test[:,0] =  (data_test[:, 1 ] / np.std(data_train[ :, 0 ]))**2
noise_var_test[:,3] =  (data_test[:, 5 ] / np.std(data_train[ :, 4 ]))**2
noise_var_test[:,4] =  (data_test[:, 7 ] / np.std(data_train[ :, 6 ]))**2

n_classes = np.max(y_train) + 1

# We estimate the log length scales

X_sample = X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = 1000), :  ]
dist2 = np.sum(X_sample**2, 1, keepdims = True) - 2.0 * np.dot(X_sample, X_sample.T) + np.sum(X_sample**2, 1, keepdims = True).T
log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(1000, 1) ]))

kernels = [ RBF_ARD(log_l * np.ones(X_train.shape[ 1 ]).astype(np.float32), -20.0, 1.0) for k in range(n_classes) ] 

# We choose the inducing points at random (different for each class)

inducing_points = np.stack([ X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = 100), \
        :  ].astype(np.float32) for i in range(n_classes) ])

model = MGPC(kernels, inducing_points, n_classes, X_train.shape[ 0 ])

np.random.seed(0)

model.fit(X_train, y_train, tf.train.AdamOptimizer(learning_rate = 0.001), 750, 50, X_test, y_test, "output_results_" + str(fold) + ".txt")

result = model.predict(X_train)
train_error = np.mean(result[ 1 ] != y_train[ : , 0 ]) 
train_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_train.shape[ 0 ]), y_train[ :, 0 ]) ]))

result = model.predict(X_test)
test_error = np.mean(result[ 1 ] != y_test[ : , 0 ]) 
test_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_test.shape[ 0 ]), y_test[ :, 0 ]) ]))

print('Train Error: %g\n' % (train_error))
print('Train NLL: %g\n' % (train_nll))
print('Test Error: %g\n' % (test_error))
print('Test NLL: %g\n' % (test_nll))


