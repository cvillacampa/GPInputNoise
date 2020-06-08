import numpy as np
import tensorflow as tf
import time
import sys

class MGPC_VI(object):

    '''
    Variational Inference multiclass classification with gaussian processes
    '''
    
    def __init__(self, kernels, inducing_points, n_classes, total_observed_x, input_var_ini, eps = 0.001):

        '''
        Constructor
        @param kernels: Kernel (covariance) functions
        @param inducing_points: Initial values for the inducing locations (Should have dimension #K, M, D or M, D)
        @param n_classes : Number of classes in the multi-class problem
        @param total_training_data : Use to scale the objective
        @param eps: eps parameter of VI
        '''

        assert n_classes > 2
        
        self.kernels = kernels
        self.n_classes = n_classes
        self.eps = eps
        total_training_data = total_observed_x.shape[0]
        self.total_training_data = tf.constant([ 1.0 * total_training_data ], dtype = tf.float32)
        self.n_points_grid = 200


        # Repeat the inducing inputs for all latent processes if we haven't been given individually
        # specified inputs per process.

        if inducing_points.ndim == 2:
            inducing_points = np.tile(inducing_points[ np.newaxis, :, : ], [ n_classes, 1, 1 ])

        # We assume the same number of inducing points per process
        
        self.num_inducing = inducing_points.shape[ 1 ]
        self.input_dim = inducing_points.shape[ 2 ]
        self.variance_prior_over_data = 1000.0

        self.train_indices = tf.placeholder(tf.int32, shape =  [None] , name = "train_indices")

        # Define placeholder variables for training and predicting.

        self.noisy_train_inputs = tf.placeholder(tf.float32, shape = [ None, self.input_dim ], name = "noisy_train_inputs")

        self.log_inputs_variances = tf.Variable(np.ones([ 1, self.input_dim ]) * np.log(input_var_ini), dtype = tf.float32)
        self.train_inputs_variances = tf.matmul(tf.ones([ tf.shape(self.noisy_train_inputs)[ 0 ], 1]), tf.exp(self.log_inputs_variances))

        self.train_outputs = tf.placeholder(tf.int32, shape = [ None, 1 ], name = "train_outputs")
        self.noisy_test_inputs = tf.placeholder(tf.float32, shape = [ None, self.input_dim ], name = "noisy_test_inputs")
        self.test_inputs_variances = tf.matmul(tf.ones([ tf.shape(self.noisy_test_inputs)[ 0 ], 1]), tf.exp(self.log_inputs_variances))
        self.test_outputs = tf.placeholder(tf.int32, shape = [ None, 1 ], name = "test_outputs")

        self.train_inputs = self.noisy_train_inputs

        self.test_inputs = self.noisy_test_inputs
        self.class_predict_test = tf.placeholder(tf.int32, shape = [ 1 ]) # This is used for making test predictions

        # Define all parameters that get optimized (one pear each class)

        self.inducing_points = tf.Variable(inducing_points, dtype = tf.float32)
        self.kernel_params = sum([ k.get_params() for k in self.kernels ], [])
        self.m = tf.Variable(tf.zeros([ self.n_classes , self.num_inducing, 1 ]))

        # We initialize the matrices L to the cholesky decomposition of the Covariance matrix of the prior

        self.Kmm = tf.stack([ self.kernels[ i ].kernel(self.inducing_points.initialized_value()[ i, :, : ]) for i in range(self.n_classes) ])
        self.chol_Kmm = tf.cholesky(self.Kmm)

        self.Lraw = tf.Variable(self.chol_Kmm - tf.matrix_band_part(self.chol_Kmm, 0, 0) + \
            tf.matrix_diag(tf.log(tf.matrix_diag_part(self.chol_Kmm))), dtype = tf.float32)

        # We transform the variables to work with the lower triangular part of the covariance matrices (exponentiating the diagonal)

        self.L = tf.matrix_band_part(self.Lraw, -1, 0) - tf.matrix_band_part(self.Lraw, 0, 0) + \
            tf.matrix_diag(tf.exp(tf.matrix_diag_part(self.Lraw)))

    
        # Now build our computational graph for training and making predictions.

        [ self.nelbo, self.test_probs ] = self._build_graph()

        # We get a tensorflow session 

        self.session = tf.Session(config = tf.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1))

        self.session.run(tf.variables_initializer(self.kernel_params))

        self.session.run(tf.global_variables_initializer())


    def hermgauss(self, n):

        # This has been extracted from GP flow. Return the locations and weights of GH quadrature

        x, w = np.polynomial.hermite.hermgauss(n)

        return x.astype(np.float32), w.astype(np.float32)

    def get_params(self):

        '''
        Gets all the parameters to be optimized
        '''

        return [ self.m, self.Lraw, self.kernel_params ]
        
    def _build_graph(self):

        '''
        Builds the computational graph
        '''
       
        KL_sum = self._build_KL_objective()
        expected_log_joint = self._build_expected_log_joint(self.train_inputs, self.train_outputs, self.train_inputs_variances)

        n_train = tf.shape(self.train_inputs)[ 0 ]

        nelbo = -1.0 * (tf.reduce_sum(expected_log_joint) * (self.total_training_data / tf.cast(n_train, tf.float32)) - KL_sum)

        test_probs = self._build_prediction_probs()

        return [ nelbo, test_probs ]
 
    def _build_KL_objective(self):

        KL_sum = 0.5 * tf.reduce_sum(- self.num_inducing + 
            2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.chol_Kmm)) - \
            tf.log(tf.matrix_diag_part(self.L)), 1) + \
            tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.chol_Kmm, self.m)), [ 1, 2 ]) + \
            tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.chol_Kmm, self.L)), [ 1, 2 ]))

        return KL_sum
        
    def _compute_probs_gh(self, data_inputs, data_targets, data_input_variances):

        '''
        Computes the probability of each point being assigned a particlar class given by the targets
        '''
 
        # We get the marginals of the values of the process for each class

        Knm = tf.stack([ self.kernels[ i ].kernel(data_inputs, self.inducing_points[ i, :, : ]) for i in range(self.n_classes) ])
        means = tf.transpose(tf.matmul(Knm, tf.cholesky_solve(self.chol_Kmm, self.m))[ :, :, 0 ])
         
        marginal_variances = tf.stack([ self.kernels[ i ].get_var_points(data_inputs) for i in range(self.n_classes) ])
        
        KmmInvKmn = tf.cholesky_solve(self.chol_Kmm, tf.transpose(Knm, [ 0, 2, 1]))
        LtKmmInv = tf.matmul(tf.transpose(self.L, [ 0, 2, 1 ]), KmmInvKmn)

        variances = tf.transpose(- tf.reduce_sum(KmmInvKmn * tf.transpose(Knm, [ 0, 2, 1 ]), 1) + \
                tf.reduce_sum(LtKmmInv * LtKmmInv, 1) + marginal_variances)

        # We compute the gradient of the mean w.r.t. the input ponit

        for i in range(self.n_classes):

            grad_posterior_mean = tf.gradients(means[ :, i ], data_inputs)[ 0 ]
            variances += tf.reduce_sum(grad_posterior_mean**2 * data_input_variances, axis = 1, keep_dims = True) * tf.one_hot(tf.ones([tf.shape(data_inputs)[ 0 ]], dtype = tf.int32) * i, self.n_classes)

        # Part of the next code is extracted from GP flow and uses gauss hermite quadrature (see wikipedia article)

        gh_x, gh_w = self.hermgauss(self.n_points_grid)
        oh_on = tf.cast(tf.one_hot(tf.reshape(data_targets, (-1,)), self.n_classes, 1., 0.), tf.float32)
        mu_selected = tf.reduce_sum(oh_on * means, 1)
        var_selected = tf.reduce_sum(oh_on * variances, 1)
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1))
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(means, 2)) / tf.expand_dims(tf.sqrt(tf.clip_by_value(variances, 1e-10, np.inf)), 2)
        normal = tf.contrib.distributions.Normal(loc = 0.0, scale = 1.0)
        log_cdfs = normal.log_cdf(dist)
        oh_off = tf.cast(tf.one_hot(tf.reshape(data_targets, (-1,)), self.n_classes, 0., 1.), tf.float32)
        log_cdfs = log_cdfs * tf.expand_dims(oh_off, 2)

        return tf.matmul(tf.exp(tf.reduce_sum(log_cdfs, reduction_indices=[1])), tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1)))

    def _build_expected_log_joint(self, data_inputs, data_outputs, data_input_variances):

        probs = self._compute_probs_gh(data_inputs, data_outputs, data_input_variances)
        
        ret = tf.log(1.0 - self.eps) * probs + (1.0 - probs) * tf.cast(tf.log(self.eps / (self.n_classes - 1)), dtype=tf.float32)
 
        return ret

    def _build_prediction_probs(self):

        data_targets = self.class_predict_test * tf.ones([ tf.shape(self.test_inputs)[ 0 ], 1 ], tf.int32)

        probs = self._compute_probs_gh(self.test_inputs, data_targets, self.test_inputs_variances)

        probs_final = (1.0 - self.eps) * probs + (1.0 - probs) * self.eps / (self.n_classes - 1)

        return probs_final


    def fit(self, training_points, training_targets, train_input_vars, optimizer, epochs, batch_size, test_points, test_targets, test_input_vars, file_results):

        """
        Fit the Gaussian process model to the given data.

        Parameters
        ----------
        training_points : Numpy array
            The training inputs
        training_targets: Numpy array of integers (1, 2, 3, ...)
            The training outputs
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        epochs : int
            The number of epochs to optimize the model for.
        batch_size : int
            The number of datapoints to use per mini-batch when training. If batch_size is None,
            then we perform batch gradient descent.
        """
    
        if batch_size is None:
            batch_size = training_points.shape[ 0 ]

        assert batch_size <= training_points.shape[ 0 ]

        self.batch_size = batch_size

        # Generate variables and operations for the minimizer and initialize variables

        self.train_step = optimizer.minimize(self.nelbo)
        self.session.run(tf.global_variables_initializer()) # no necesario

        n_batches = int(np.ceil(float(training_points.shape[ 0 ]) / self.batch_size))
        ini_time = time.time()
        
        original_indices = np.arange(0, training_points.shape[ 0 ])
 
        for e in range(epochs):

            avg_elbo = 0.0
            permutation = np.random.permutation(training_points.shape[ 0 ])
            original_indices = original_indices[ permutation ]
            training_points = training_points[ permutation, : ]
            training_targets = training_targets[ permutation, : ]

            start = time.time()

            for i in range(n_batches):

                train_indices = np.arange(i * self.batch_size , np.minimum((i + 1) * self.batch_size, training_points.shape[ 0 ])).astype(np.int32) 
                
                batch_x = training_points[ train_indices, : ].astype(np.float32)
                batch_y = training_targets[ train_indices, : ].astype(np.int32)

                avg_elbo += -1.0 * self.session.run(self.nelbo, feed_dict = \
                    {self.noisy_train_inputs: batch_x, self.train_outputs: batch_y, \
			        self.train_indices: original_indices[ train_indices ]})

                self.session.run(self.train_step, feed_dict = \
                    {self.noisy_train_inputs: batch_x, self.train_outputs: batch_y, \
			        self.train_indices: original_indices[ train_indices ]})

            end = time.time()

            print('Epoch %d, Avg. elbo %g, Time %g' % (e, avg_elbo / n_batches, end - start))
	    sys.stdout.flush()

        print('Input noise variances:')
        print(self.session.run(tf.exp(self.log_inputs_variances)))

	ini_time = self.write_results(test_points, test_targets, test_input_vars, file_results, ini_time, test_targets.shape[ 0 ])

    def write_results(self, data_points, data_targets, input_vars, file_name, ini_time, batch_size = None):

        start_predict = time.time()
        result = self.predict(data_points, input_vars, batch_size)
        end_predict = time.time()

        ini_time += end_predict - start_predict

        error = np.mean(result[ 1 ] != data_targets[ : , 0 ]) 
        nll = -np.mean(np.log(result[ 0 ][ (np.arange(data_points.shape[ 0 ]), data_targets[ :, 0 ]) ]))
        

        with open("results/" + file_name, "a") as myfile:
            myfile.write(str(error) + " " + str(nll) + " " + str(time.time() - ini_time) + '\n')

        return ini_time

    def predict(self, test_inputs, test_input_vars, batch_size = None):

        """
        Compute predictions for test data.

        Parameters
        ----------
        test_inputs : Numpy array
            The test inputs
 
        """

        if batch_size is None:
            batch_size = self.batch_size

        probs = np.zeros((test_inputs.shape[ 0 ], self.n_classes))

        n_batches = int(np.ceil(float(test_inputs.shape[ 0 ]) / batch_size))
        
        for i in range(n_batches):
            for c in range(self.n_classes):
                indices = np.arange(i * batch_size, np.minimum((i + 1) * batch_size, test_inputs.shape[ 0 ]))
                
                probs[ i * batch_size : np.minimum((i + 1) * batch_size, test_inputs.shape[ 0 ]), c : (c + 1) ] = \
                    self.session.run(self.test_probs, feed_dict = {self.noisy_test_inputs: test_inputs[ indices , : ], \
		                             self.test_inputs_variances: test_input_vars[ indices, : ], self.class_predict_test: np.array([ c ])})

        return [ probs, np.argmax(probs, 1) ] 


