import numpy as np
import tensorflow as tf
import sys
sys.path.append("../../../..")
from gp_input_noise.mgpc import MGPC
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

## Data Augmentation Sampling from posterior assuming uniform prior
## p(mu|x) propto N(x|mu,sigma) * p(mu) propto N(x|mu,sigma) = N(mu|x,sigma)

n_samples = 2

X_train_augmented = X_train
y_train_augmented = y_train

for i in range(2):

    state = np.random.get_state()
    np.random.seed(fold * 1000 + i)

    new_data_x = X_train + np.random.normal(size = X_train.shape) * np.sqrt(2.0 * noise_var_train)
    new_data_y = y_train.copy()
    X_train_augmented = np.vstack((X_train_augmented, new_data_x))
    y_train_augmented = np.vstack((y_train_augmented, new_data_y))

    np.random.set_state(state)

X_train = X_train_augmented.copy()
y_train = y_train_augmented.copy()

state = np.random.get_state()
np.random.seed(fold * 2000 + i)
perm = np.random.permutation(X_train.shape[ 0 ])
X_train = X_train[ perm, : ]
y_train = y_train[ perm, : ]
np.random.set_state(state)

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

model.fit(X_train, y_train, tf.train.AdamOptimizer(learning_rate = 0.01), 750, 200, X_test, y_test, "output_results_" + str(fold) + ".txt")

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


