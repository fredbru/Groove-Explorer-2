import numpy as np
from collections import defaultdict
from warnings import warn
from numba import cuda
import math
import timeit
#import cvxpy as cvx
#import dccp

""" Self-Organising Map library by Fred Bruford
    Functionality for both GPU and CPU computation. CPU computation will work on any machine, GPU should but may have
    issues (Tested on NVIDIA Tesla P100).

    HIGHLY RECOMMENDED: If using GPU, run on an external server/machine. If you try to use your built in GPU that's
    used for processing your display, you can run into issues with CUDA blocking processing times over a few seconds.
"""

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array for CPU computation.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return math.sqrt(np.dot(x, x.T))
    #return np.linalg.norm(x)

@cuda.jit
def GPUSOM(weights, activation_map, data, ax, ay,g,iterations,random,randomItem,perceptualWeighting):
    """
    Big CUDA kernel function for training SOM on a GPU. Currently runs x dimension of the SOM in parallel.
    Ideally should run each node in it's own thread, but currently limited by memory on one GPU. Future work to run
    across a GPU cluster could fix this. For small SOMs, this will still work, so kept commented out code in.
    Essentially 3 stages - find winner node, generate radius/learning rate function, apply function to weights.
    :param weights: initialised SOM weights
    :param activation_map:
    :param data: feature data
    :param ax:
    :param ay:
    :param g:
    :param iterations: number of iterations
    :param random:
    :param randomItem:
    :param perceptualWeighting:
    :return:
    """
    startx = cuda.grid(1)
    stridex = cuda.gridsize(1)
    # startx,starty = cuda.grid(2) # include thsese lines for 2D multithreading (across 2 dimensions of nodes).
    # stridex, stridey = cuda.gridsize(2)

    squaredSum = 0.0
    sigma = 0.3
    learning_rate = 0.5
    pi = np.pi
    norm=0.0

    for iteration in range(iterations):
        if startx<weights.shape[0]:
            #print(iteration)
            # pick random item
            randomItem = data[random[iteration, 0]]
            # Calculate winner (equivalent to self.winner, which calls self.activate)
            # Subtract item from each node, then find norm for each node.
            # Activation map is this norm - kinda like a 'fitness' score for each item.
            for i in range(startx, weights.shape[0], stridex):
                for j in range(weights.shape[1]):
                # for j in range(starty, weights.shape[1],stridey):
                    activationSum = 0.0
                    for k in range(weights.shape[2]):
                        # itemDifference[k] = randomItem[k] - weights[i,j,k]
                        activationSum = activationSum + ((randomItem[k] - weights[i, j, k])
                                                         * (randomItem[k] - weights[i, j, k]))
                    activation_map[i, j] = math.sqrt(activationSum)

            cuda.syncthreads()
            ncol = activation_map.shape[1]
            winner = divmod(activation_map.argmin(),ncol)   # get winner as coordinates of SOM

        # create gaussian decay function around winner node, with learning rate
            decay = (1.0 + iteration/(iterations/2.0))
            sig = sigma / decay # sig = radius function matrix. 0.7 = sigma value
            eta = learning_rate / decay
            d = 2.0 * pi * pow(sig,2)
            for i in range(ax.shape[0]):
                ax[i] = math.exp(-pow(i-winner[0],2)/d)
                ay[i] = math.exp(-pow(i-winner[1],2)/d)

            #np.outer(ax, ay) rewritten for numba compilation, with learning rate
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    g[i, j] = ax[i] * ay[j] * eta

            for i in range(startx, weights.shape[0],stridex):
                for j in range(weights.shape[1]):
                #for j in range(starty, weights.shape[1],stridey):

                    squaredSum = 0.0
                    for k in range(weights.shape[2]):
                        # add differences * radius constant to all items
                        #x_w[i,j,k] = (randomItem[k] - weights[i,j,k])
                        weights[i,j,k] = weights[i,j,k] + (g[i,j] * (randomItem[k] - weights[i,j,k]))
                    for k in range(weights.shape[2]):
                        # get squared sum for norm calculation
                        squaredSum = squaredSum + (weights[i,j,k]*weights[i,j,k])
                    norm = math.sqrt(squaredSum)
                    #normalise weights
                    #norm = weights[i,j]=math.sqrt(np.sum(np.square(weights[i,j])))
                    for k in range(weights.shape[2]):
                        #divide weights by norm
                        weights[i,j,k] = weights[i,j,k]/norm
            cuda.syncthreads()

class MusicSOM(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                  perceptualWeighting=True, random_seed=11, coefficients_file='BIG_Coefficients.npy'):

        """Initializes a Self Organizing Map.
        Parameters
        ----------
        decision_tree : decision tree
        The decision tree to be exported.
        x : int
            x dimension of the SOM
        y : int
            y dimension of the SOM
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
            learning_rate, initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)
        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            default function:
            lambda x, current_iteration, max_iter :
                        x/(1+current_iteration/max_iter)
        neighborhood_function : function, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map
            possible values: 'gaussian', 'mexican_hat'
        random_seed : int, optional (default=None)
            Random seed to use.
        """
        self._random_generator = np.random.RandomState(random_seed)
        if sigma >= x / 2.0 or sigma >= y / 2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        self.learning_rate = learning_rate
        self.sigma = sigma
        # random initialization
        self.weights = self._random_generator.rand(x, y, input_len)
        # continue from previous
        #self.weights = np.load("SOM_Weights.npy")
        #print("continued from 1M")
        self.activation_map = np.zeros((x, y))

        self._neigx = np.arange(x)
        self._neigy = np.arange(y)  # used to evaluate the neighborhood function

        self.wL = np.ones(input_len, dtype=np.float64)
        self.SOMSize = x * y
        self.x = x
        self.y = y
        self.perceptual_weights = np.full(input_len, 0.5)
        #self.local_weights = np.ones(self.weights.shape)
        path = "Groove-Explorer-2/"
        coefficients = np.load(path+coefficients_file).reshape(1,1,input_len)

        self.local_weights = np.broadcast_to(coefficients, (x,y,input_len))
        self.initial_weights = np.copy(self.local_weights)
        self.active_learning_step_count = 0
        self.local_weights_history =  []
        self.local_weights_history.append(np.copy(self.local_weights))

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x
        Done on CPU.
        :param x: sample item
        """
        #s = np.subtract(self.perceptual_weights * x, self.weights * self.perceptual_weights)
        s = np.subtract(x*self.local_weights, self.weights*self.local_weights)  # x - w #NEED TO CHANGE THIS TO x*perceptual weights for global weighting
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            # || x - w ||
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()

        winner = np.unravel_index(self.activation_map.argmin(),
                               self.activation_map.shape)
        return winner

    def trainGPU(self, dataCPU, num_iteration):
        """
        Batch trains the SOM on a GPU. Moves data to GPU and invokes CUDA kernel
        :param dataCPU: feature data
        :param num_iteration: number of training iterations
        :return:
        """
        #print("Warp size = ", device.WARP_SIZE)
        n = 64000
        threadsperblock = self.x # create x*y threads on gpu - thread for each SOM node.

        # Generate random numbers on CPU and transport in one go (easier than generating them on GPU)
        randomList = cuda.to_device(np.random.randint(0,dataCPU.shape[0],size=(num_iteration,1)).astype('i'))
        randomItem = cuda.to_device(np.empty_like(dataCPU[1]).astype('f'))

        # move data to GPU
        startDataTransfer = timeit.default_timer()
        GPUweights = cuda.to_device(self.weights.astype('f'))
        GPUactivationMap = cuda.to_device(self.activation_map.astype('f'))
        GPUInputData= cuda.to_device(dataCPU.astype('f'))
        ax = cuda.to_device(np.empty_like(self._neigx).astype('f'))
        ay = cuda.to_device(np.empty_like(self._neigy).astype('f'))
        g = cuda.to_device(np.empty([self.x, self.y]).astype('f'))
        perceptualWeighting = cuda.to_device(self.perceptualWeightVector.astype('f')) #this isn't working
        cuda.synchronize()
        endDataTransfer = timeit.default_timer()


        startTraining = timeit.default_timer()

        GPUSOM[128,threadsperblock](GPUweights,GPUactivationMap ,GPUInputData,ax, ay, g, num_iteration, randomList,
                                    randomItem,perceptualWeighting)
        endTraining = timeit.default_timer()
        print("GPU Training time = " + str(endTraining-startTraining))


        cuda.synchronize()
        startDataTransfer2 = timeit.default_timer()
        self.weights=GPUweights.copy_to_host()
        endDataTransfer2 = timeit.default_timer()
        totalDataTransferTime = (endDataTransfer-startDataTransfer)+(endDataTransfer2-startDataTransfer2)
        print("GPU data transfer time = " + str(totalDataTransferTime))

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data"""
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self.weights[it.multi_index] = data[rand_i]
            norm = fast_norm(self.weights[it.multi_index])
            self.weights[it.multi_index] = self.weights[it.multi_index] / norm
            it.iternext()

    def trainCPU(self, data, num_iterations):
        """
        Batch trains the SOM on a CPU.
        :param data: feature data
        :param num_iteration: number of training iterations
        :return:
        """
        for iteration in range(num_iterations):
            # pick a random sample
            rand_i = self._random_generator.randint(len(data))
            randomItem = data[rand_i]
            winner = self.winner(randomItem)
            g = self.getUpdateFunction(winner, iteration, num_iterations)
            self.update(randomItem, iteration, g, num_iterations)
        print(self.weights)

    def getUpdateFunction(self, winner, iteration, num_iterations):
        """ Generate update function for node matrix, considering winner location, radius function and learning rate
        :param winner: coordinates of winner node
        :param iteration: iteration number
        :param num_iterations: total iterations
        :return:
        """
        decay = (1.0 + iteration / (num_iterations / 2.0))
        sig = self.sigma / decay  # sig = radius function matrix. 0.7 = sigma value
        eta = self.learning_rate / decay
        d = 2.0 * np.pi * pow(sig, 2)
        ax = np.exp(-np.power(self._neigx - winner[0], 2) / d)
        ay = np.exp(-np.power(self._neigy - winner[1], 2) / d)

        g = np.outer(ax, ay) *eta
        return g

    def update(self, x, iteration, g, num_iterations):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        """
        it = np.nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            x_w = (x - self.weights[it.multi_index])
            self.weights[it.multi_index] += g[it.multi_index] * x_w
            # normalization
            norm = fast_norm(self.weights[it.multi_index])
            self.weights[it.multi_index] = self.weights[it.multi_index] / norm
            it.iternext()

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours."""
        um = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                    if (ii >= 0 and ii < self.weights.shape[0] and
                                jj >= 0 and jj < self.weights.shape[1]):
                        w_1 = self.weights[ii, jj, :]
                        w_2 = self.weights[it.multi_index]
                        um[it.multi_index] += fast_norm(w_1 - w_2)
            it.iternext()
        um = um / um.max()
        return um

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c"""
        # d = 2 * np.pi * sigma * sigma
        # ax = np.exp(-np.power(self._neigx - c[0], 2) / d)
        # ay = np.exp(-np.power(self._neigy - c[1], 2) / d)
        # return np.outer(ax, ay)  # the external product gives a matrix

        sig = self.sigma
        d = 2.0 * np.pi * pow(sig, 2)
        ax = np.exp(-np.power(self._neigx - c[0], 2) / d)
        ay = np.exp(-np.power(self._neigy - c[1], 2) / d)

        g = np.outer(ax, ay)
        return g

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        error = 0
        for x in data:
            error += fast_norm(x - self.weights[self.winner(x)])
        return error / len(data)

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

    # def update_active_learning_quadratic(self, groove, new_coordinates, old_coordinates):
    #     # Perform active learning from a single movement
    #     old_weights = self.perceptual_weights
    #     q = -old_weights
    #     P = np.identity(groove.shape[0])
    #
    #     #A = (groove-self.weights[old_coordinates[0],old_coordinates[1],:]) - \
    #     #    (groove-self.weights[new_coordinates[0],new_coordinates[1],:])
    #     #print('A= ', A)
    #
    #     # need to change this constraint - for d1 to not just be larger than d2 but larger than
    #     # all other distancess
    #     x = cvx.Variable(groove.shape[0], nonneg=True)
    #
    #     target_d = groove-self.weights[new_coordinates[0],new_coordinates[1],:]
    #     target_d_sum = cvx.norm(cvx.multiply(target_d, x))
    #     constraints = []
    #     node_d = []
    #     node_d_sum = []
    #     it = np.nditer(self.activation_map, flags=['multi_index'])
    #     i=0
    #     while not it.finished:
    #         if [it.multi_index[0], it.multi_index[1]] != [new_coordinates[0], new_coordinates[1]]:
    #             node_d.append(groove - self.weights[it.multi_index[0],
    #                                                     it.multi_index[1], :])
    #             node_d_sum.append(cvx.norm(cvx.multiply(node_d[i], x)))
    #             #print("node d sum ", i, " is DCP? = ", node_d_sum[i].is_dcp())
    #             #print("node d curve ", node_d_sum[i].curvature)
    #             #print("target d curve ", target_d_sum.curvature)
    #
    #             diff = node_d_sum[i] - target_d_sum
    #
    #             #print("difference curve ", diff.curvature)
    #
    #             #print('node d sum', node_d_sum[i].size, node_d_sum[i].is_scalar())
    #             #print("Diff is scalar? ", diff.is_scalar())
    #             #constraints += [(-1.0 * node_d_sum[i]) >= (-1.0*target_d_sum)]
    #             #constraints += [diff >= 0.0]
    #             constraints += [node_d_sum[i] >= target_d_sum] # need to do 2 norm..?
    #             # print("constraints = dcp? ", constraints[i].is_dcp())
    #      #       constraints += [cp.sum(node_distance[i] @ x) >= cp.sum(target_d @ x)] # need to do 2 norm..?
    #             i+=1
    #         it.iternext()
    #     constraints+=[0.00<=x]
    #     constraints+=[x<=1.0]
    #     problem = cvx.Problem(cvx.Minimize((1/2)*cvx.quad_form(x,P) + q.T @ x),constraints)
    #                          # [ds@x >= d2@x,
    #                          #  #A @ x <= u,
    #                          #  0.00001 <= x,
    #                          #  x <= 1])
    #     print("Problem = DCCP? ", dccp.is_dccp(problem))
    #     problem.solve(method='dccp', max_iter=100)
    #     print("status:", problem.status)
    #     print("optimal value", problem.value)
    #     new_x = x.value
    #     print("x = ", new_x)
    #     self.perceptual_weights = new_x
    #     print('result =', self.winner(groove))
    #     # need to incorporate perceptual weighting into som map

    def update_active_learning_nurnberger_global(self, groove, new_coordinates, old_coordinates):
        source_node = self.weights[old_coordinates[0],old_coordinates[1],:]
        target_node = self.weights[new_coordinates[0],new_coordinates[1],:]

        source_error_vector = np.absolute(groove-source_node / fast_norm(groove - source_node))
        target_error_vector = np.absolute(groove-target_node / fast_norm(groove - target_node))

        error_vector = source_error_vector - target_error_vector # = f
        learning_rate = 0.00005
        new_x = new_coordinates[0]
        possible_x = new_x, new_x + 1, new_x -1
        new_y = new_coordinates[1]
        possible_y = new_y, new_y + 1, new_y -1

        for i in range(100000):
            #print(i)
            winner = self.winner(groove)
            winner_x = winner[0]
            winner_y = winner[1]
            if (winner_x in possible_x) and (winner_y in possible_y):
                #print("Winner = ", self.winner(groove))
                break
            else:
                self.perceptual_weights = self.perceptual_weights +(error_vector * learning_rate)
        print(self.perceptual_weights)

    def update_active_learning_nurnberger_local(self, groove, new_coordinates, old_coordinates):

        source_node = self.weights[old_coordinates[0], old_coordinates[1], :]
        target_node = self.weights[new_coordinates[0], new_coordinates[1], :]

        source_error = np.absolute(groove - source_node / fast_norm(groove - source_node)).reshape(1,1,groove.shape[0])
        target_error = np.absolute(groove - target_node / fast_norm(groove - target_node)).reshape(1,1,groove.shape[0])

        new_x = new_coordinates[0]
        possible_x = new_x, new_x + 1, new_x -1 #, new_x + 2, new_x - 2
        new_y = new_coordinates[1]
        possible_y = new_y, new_y + 1, new_y -1 #, new_y + 2, new_y -2
        #print('n')
        sigma = 16.0
        source_distance_map = self._gaussian(old_coordinates, sigma).reshape(12,12,1)
        target_distance_map = self._gaussian(new_coordinates, sigma).reshape(12,12,1)

        learning_rate = 0.001
        for i in range(20000):
            winner = self.winner(groove)
            winner_x = winner[0]
            winner_y = winner[1]
            if (winner_x in possible_x) and (winner_y in possible_y):
                # print("Winner = ", self.winner(groove))
                break
            else:
                # local_weights = x,y,input_len
                # source_error = input_len
                # source/target_distance_map = x,y
                old_weights = self.local_weights.copy()
                x = old_weights + (learning_rate * ((source_error*source_distance_map) - (target_error*target_distance_map)))
                self.local_weights =  x / np.max(x)
                # print("localweights= ", self.local_weights[new_x, new_y, :])

        self.active_learning_step_count += 1
        self.local_weights_history.append(np.copy(self.local_weights))

        for i in range(2):
            randomItem = groove
            winner = self.winner(randomItem)
            g = self.getUpdateFunction(winner, 1,1)
            self.update(randomItem, 1, g, 1)


    def revert_active_learning(self):
        if self.active_learning_step_count > 0:
            self.local_weights = self.local_weights_history[self.active_learning_step_count - 1]
            del(self.local_weights_history[self.active_learning_step_count])
            self.active_learning_step_count -= 1

    def reset_active_learning(self):
        self.local_weights = np.copy(self.initial_weights)
        del self.local_weights_history
        self.local_weights_history =  []
        self.local_weights_history.append(self.local_weights)
        self.active_learning_step_count = 0

