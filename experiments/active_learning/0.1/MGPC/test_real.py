import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append("../../../..")
from gp_input_noise.mgpc import MGPC
from gp_input_noise.rbf_ard import RBF_ARD


if len(sys.argv) != 3:
        print("Incorrect number of parameters:\n\tUse: {} <fold> random|min_ll".format(sys.argv[0]))
        exit(1)

fold = int(sys.argv[1])
mode = sys.argv[2]

if mode != "random" and mode != "min_ll":
        print("Incorrect selection mode. Must be random|min_ll")
        exit(2)

X = np.load("../../data/X.npy")
Y = np.load("../../data/Y.npy").astype(np.int32)

noise = np.load("../../data/noise01.npy")[ fold, : ]

# We set the seed to the number of split and by shuffling the data we get a different split

np.random.seed(fold)

sample = np.random.choice(X.shape[0], X.shape[0])

X = X[sample, :]
Y = Y[sample]

# We select the initial train/test/validation partition

n_train = 100
n_test = 500
n_val = 400
n_to_add = 100

X_train = X[ 0 : n_train, ]
y_train = Y[ 0 : n_train ]
X_test = X[ n_train : (n_train + n_test), ]
y_test = Y[ n_train : (n_train + n_test) ]
X_val = X[ (n_train + n_test) : (n_train + n_test + n_val), ]
y_val = Y[ (n_train + n_test) : (n_train + n_test + n_val) ]

y_train = y_train[:, None]
y_test = y_test[:, None]
y_val = y_val[:, None]

# We normalize the data

mean_train = np.mean(X_train, 0)
std_train = np.std(X_train, 0)

X_train = (X_train - mean_train) / std_train
X_test = (X_test - mean_train) / std_train
X_val = (X_val - mean_train) / std_train


# We inject the noise 

X_train = X_train + noise[ 0 : n_train,  : ] 
X_test = X_test + noise[ n_train : (n_train + n_test),  : ] 
X_val = X_val + noise[ (n_train + n_test) : (n_train + n_test + n_val),  : ] 

n_classes = np.max(y_train) + 1

# We estimate the log length scales

X_sample = X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = X_train.shape[0]), :  ]
dist2 = np.sum(X_sample**2, 1, keepdims = True) - 2.0 * np.dot(X_sample, X_sample.T) + np.sum(X_sample**2, 1, keepdims = True).T
log_l = 0.5 * np.log(np.median(dist2[ np.triu_indices(X_train.shape[0], 1) ]))

kernels = [ RBF_ARD(log_l * np.ones(X_train.shape[ 1 ]).astype(np.float32), -20.0, 1.0) for k in range(n_classes) ] 

# We choose the inducing points at random (different for each class)

M = 50

inducing_points = np.stack([ X_train[ np.random.choice(np.arange(X_train.shape[ 0 ]), size = M), \
        :  ].astype(np.float32) for i in range(n_classes) ])

model = MGPC(kernels, inducing_points, n_classes, X_train.shape[ 0 ])

np.random.seed(0)

# We delete previous results
if os.path.exists("results/" + mode + "_train_" + str(fold) + ".txt"):
    os.remove("results/" + mode + "_train_" + str(fold) + ".txt")
if os.path.exists("results/" + mode + "_test_" + str(fold) + ".txt"):
    os.remove("results/" + mode + "_test_" + str(fold) + ".txt")

# We pretrain for 1000 epochs

model.fit(X_train, y_train, tf.train.AdamOptimizer(learning_rate = 0.001), 1000, 50, X_test, y_test, mode + "_output_results_" + str(fold) + ".txt")


# We pass through all the data adding a new point each time
# We add n_to_add points during training
for i in range(n_to_add):
       
        # Evaluate log_likelihood

        result = model.predict(X_train)
        train_error = np.mean(result[ 1 ] != y_train[ : , 0 ]) 
        train_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_train.shape[ 0 ]), y_train[ :, 0 ]) ]))

        result = model.predict(X_test)
        test_error = np.mean(result[ 1 ] != y_test[ : , 0 ]) 
        test_nll = -np.mean(np.log(result[ 0 ][ (np.arange(X_test.shape[ 0 ]), y_test[ :, 0 ]) ]))

        print("Adding new point...error: {} nll: {}".format(test_error, test_nll))

        # Save results

        with open("results/" + mode + "_train_" + str(fold) + ".txt", "a") as myfile:
            myfile.write(str(train_error) + " " + str(train_nll) + '\n')

        with open("results/" + mode + "_test_" + str(fold) + ".txt", "a") as myfile:
            myfile.write(str(test_error) + " " + str(test_nll) + '\n')


        # Add new point
        if mode == "random":
                # Select the next point in the validation set
                index_new_point = i
        elif mode == "min_ll":
                index_new_point = i

                # Select the point with more uncertainty in the prediction
                result = model.predict(X_val)

                entropy = -np.sum(result[0] * np.log(result[0]), 1)
                index_new_point = np.argmax(entropy)


        # Add new point to training set

        X_train = np.concatenate([X_train, X_val[index_new_point, :][None,: ]], 0)
        y_train = np.concatenate([y_train, y_val[index_new_point, :][None,: ]], 0)
        
        # Remove data point from Xval

        X_val = np.delete(X_val, [index_new_point], 0)
        y_val = np.delete(y_val, [index_new_point], 0)
        
        # Renormalize all training set with new point

        mean_train = np.mean(X_train, 0)
        std_train = np.std(X_train, 0)

        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        X_val = (X_val - mean_train) / std_train

        n_train = n_train + 1

        # Retrain for 25 epochs

        model.fit(X_train, y_train, None, 1000, 50, X_test, y_test, mode + "_output_results_" + str(fold) + ".txt", continue_training=True)
