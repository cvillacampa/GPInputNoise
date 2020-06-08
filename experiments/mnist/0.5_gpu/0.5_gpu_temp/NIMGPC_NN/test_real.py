import numpy as np
import tensorflow as tf
import sys
sys.path.append("../../../..")
from gp_input_noise.nimgpc_nn_learn_noise import NIMGPC_NN
from gp_input_noise.rbf_ard import RBF_ARD

np.random.seed(0)

fold = 0

X = np.load("../../data/X.npy")
Y = np.load("../../data/Y.npy").astype(np.int).flatten()

indices = np.load("../../data/splits.npy")[ fold, : ]
noise = np.load("../../data/noise5.npy")


n_train = int(len(indices) * 0.8571429)
indices_train = indices[:n_train]
indices_test = indices[n_train:]

X_train, y_train, X_test, y_test = X[ indices_train, : ], Y[ indices_train ], X[ indices_test, : ], Y[ indices_test ]
y_train = y_train[:, None]
y_test = y_test[:, None]

mean_train = np.mean(X_train, 0)
std_train = np.std(X_train, 0)
std_train[ std_train == 0 ] = 1.0

X_train = (X_train - mean_train) / std_train
X_test = (X_test - mean_train) / std_train

X_train = X_train + noise[ indices_train,  : ] * 0.0
X_test = X_test + noise[ indices_test,  : ] * 0.0




n_classes = np.max(y_train) + 1

# We estimate the log length scales

X_sample = X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = 1000), :  ]
dist2 = np.sum(X_sample**2, 1, keepdims = True) - 2.0 * np.dot(X_sample, X_sample.T) + np.sum(X_sample**2, 1, keepdims = True).T
log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(1000, 1) ]))

kernels = [ RBF_ARD(log_l * np.ones(X_train.shape[ 1 ]).astype(np.float32), -20.0, 1.0) for k in range(n_classes) ] 

# We choose the inducing points at random (different for each class)

inducing_points = np.stack([ X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = np.minimum(100, int(X.shape[ 0 ] * 0.05))), \
        :  ].astype(np.float32) for i in range(n_classes) ])

model = NIMGPC_NN(kernels, inducing_points, n_classes, X_train, 0.5, n_layers = 2, n_hiddens = 250, soft_ini = True)

np.random.seed(0)

model.fit(X_train, y_train, tf.train.AdamOptimizer(learning_rate = 0.001), 350, 200, X_test, y_test, "output_results_" + str(fold) + ".txt")

#result = model.predict(X_train)
#train_error = np.mean(result[ 1 ] != y_train[ : , 0 ]) 
#train_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_train.shape[ 0 ]), y_train[ :, 0 ]) ]))
#
#result = model.predict(X_test)
#test_error = np.mean(result[ 1 ] != y_test[ : , 0 ]) 
#test_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_test.shape[ 0 ]), y_test[ :, 0 ]) ]))
#
#print('Train Error: %g\n' % (train_error))
#print('Train NLL: %g\n' % (train_nll))
#print('Test Error: %g\n' % (test_error))
#print('Test NLL: %g\n' % (test_nll))


