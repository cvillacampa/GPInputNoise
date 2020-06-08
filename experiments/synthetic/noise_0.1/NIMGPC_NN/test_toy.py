import numpy as np
import tensorflow as tf
import sys
sys.path.append("../../../..")
from gp_input_noise.nimgpc_nn import NIMGPC_NN
from gp_input_noise.data import load_object
from gp_input_noise.rbf_ard import RBF_ARD

np.random.seed(0)

fold = int(sys.argv[1])
data = load_object("../data/dataset_noisy_" + str(fold) + ".dat")

X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
y_train = y_train[:, None]
y_test = y_test[:, None]

noise_var_train = np.zeros_like(X_train) + 0.1
noise_var_test = np.zeros_like(X_test) + 0.1

n_classes = np.max(y_train) + 1

# We estimate the log length scales

X_sample = X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = 1000), :  ]
dist2 = np.sum(X_sample**2, 1, keepdims = True) - 2.0 * np.dot(X_sample, X_sample.T) + np.sum(X_sample**2, 1, keepdims = True).T
log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(1000, 1) ]))

kernels = [ RBF_ARD(log_l * np.ones(X_train.shape[ 1 ]).astype(np.float32), -20.0, 1.0) for k in range(n_classes) ] 

# We choose the inducing points at random (different for each class)

inducing_points = np.stack([ X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = 100), \
        :  ].astype(np.float32) for i in range(n_classes) ])

model = NIMGPC_NN(kernels, inducing_points, n_classes, X_train, n_layers = 2)

np.random.seed(0)

model.fit(X_train, y_train, noise_var_train, tf.train.AdamOptimizer(learning_rate = 0.01), 750, 200, X_test, y_test, noise_var_test, "output_results_" + str(fold) + ".txt")

result = model.predict(X_train, noise_var_train, X_train.shape[ 0 ] / 10)
train_error = np.mean(result[ 1 ] != y_train[ : , 0 ]) 
train_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_train.shape[ 0 ]), y_train[ :, 0 ]) ]))

result = model.predict(X_test, noise_var_test, X_test.shape[ 0 ] / 10)
test_error = np.mean(result[ 1 ] != y_test[ : , 0 ]) 
test_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_test.shape[ 0 ]), y_test[ :, 0 ]) ]))

print('Train Error: %g\n' % (train_error))
print('Train NLL: %g\n' % (train_nll))
print('Test Error: %g\n' % (test_error))
print('Test NLL: %g\n' % (test_nll))


