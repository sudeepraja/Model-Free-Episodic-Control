__author__ = 'sudeep raja'
import numpy as np
import cPickle
import heapq
import image_preprocessing as ip
from sklearn.neighbors import BallTree,KDTree


class LRU_KNN:
    def __init__(self, capacity,dimension_result):
        self.capacity = capacity
        self.states = np.zeros((capacity,dimension_result))
        self.q_values = np.zeros(capacity)
        self.lru = np.zeros(capacity)
        self.curr_capacity = 0
        self.tm = 0.0
        self.tree = None

    def peek(self,key,value,modify):
        if self.curr_capacity==0:
            return None

        # tree = KDTree(self.states[:self.curr_capacity])
        dist, ind = self.tree.query([key], k=1)
        ind = ind[0][0]

        if np.allclose(self.states[ind],key):
            self.lru[ind] = self.tm
            self.tm +=0.01
            if modify:
                self.q_values[ind] = max(self.q_values[ind],value)
            return self.q_values[ind]

        return None

    def knn_value(self, key, knn):
        if self.curr_capacity==0:
            return 0.0

        # tree = KDTree(self.states[:self.curr_capacity])
        dist, ind = self.tree.query([key], k=knn)

        value = 0.0
        for index in ind[0]:
            value += self.q_values[index]
            self.lru[index] = self.tm
            self.tm+=0.01

        return value / knn

    def add(self, key, value):
        if self.curr_capacity >= self.capacity:
            # find the LRU entry
            old_index = np.argmin(self.lru)
            self.states[old_index] = key
            self.q_values[old_index] = value
            self.lru[old_index] = self.tm
        else:
            self.states[self.curr_capacity] = key
            self.q_values[self.curr_capacity] = value
            self.lru[self.curr_capacity] = self.tm
            self.curr_capacity+=1
        self.tm += 0.01
        self.tree = KDTree(self.states[:self.curr_capacity])


class Node(object):
    def __init__(self, time, state, q_return):
        self.lru_time = time  # time stamp used for LRU
        self.state = state
        self.QEC_value = q_return


class DistanceNode(object):
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

class QECTable(object):
    def __init__(self, knn, state_dimension, projection_type, observation_dimension, buffer_size, num_actions, rng):
        self.knn = knn
        self.ec_buffer = []
        self.buffer_maximum_size = buffer_size
        self.rng = rng
        for i in range(num_actions):
            self.ec_buffer.append(LRU_KNN(buffer_size,state_dimension))

        # projection
        """
        I tried to make a function self.projection(state)
        but cPickle can not pickle an object that has an function attribute
        """
        self._initialize_projection_function(state_dimension, observation_dimension, projection_type)

    def _initialize_projection_function(self, dimension_result, dimension_observation, p_type):
        if p_type == 'random':
            self.matrix_projection = self.rng.randn(dimension_result, dimension_observation).astype(np.float32)
        elif p_type == 'VAE':
            pass
        else:
            raise ValueError('unrecognized projection type')

    """estimate the value of Q_EC(s,a)  O(N*logK*D)  check existence: O(N) -> KNN: O(D*N*logK)"""
    def estimate(self, s, a):
        state = np.dot(self.matrix_projection, s.flatten())

        q_value = self.ec_buffer[a].peek(state,None,modify = False)
        if q_value!=None:
            return q_value

        return self.ec_buffer[a].knn_value(state,self.knn)

    def update(self, s, a, r):  # s is 84*84*3;  a is 0 to num_actions; r is reward
        state = np.dot(self.matrix_projection, s.flatten())
        q_value = self.ec_buffer[a].peek(state,r,modify = True)
        if q_value==None:
            self.ec_buffer[a].add(state,r)


class TraceNode(object):
    def __init__(self, observation, action, reward, terminal):
        self.image = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal


class TraceRecorder(object):
    def __init__(self):
        self.trace_list = []

    def add_trace(self, observation, action, reward, terminal):
        node = TraceNode(observation, action, reward, terminal)
        self.trace_list.append(node)









