__author__ = 'frankhe'
"""
episodic control agent
"""
import time
import os
import logging
import numpy as np
import cPickle
import EC_functions


class EpisodicControl(object):
    def __init__(self, qec_table, ec_discount, num_actions, epsilon_start, epsilon_min, epsilon_decay, exp_pref, rng):
        self.qec_table = qec_table
        self.ec_discount = ec_discount
        self.num_actions = num_actions
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.exp_pref = exp_pref
        self.rng = rng

        self.trace_list = EC_functions.TraceRecorder()

        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        self.exp_dir = self.exp_pref + time_str + \
                       "{}".format(self.ec_discount).replace(".", "p")

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self._open_results_file()

        self.step_counter = None
        self.episode_reward = None

        self.total_reward = 0.
        self.total_episodes = 0

        self.start_time = None

        self.last_img = None
        self.last_action = None

        self.steps_sec_ema = 0.

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write('epoch, episode_nums, total_reward, avg_reward\n')
        self.results_file.flush()

    def _update_results_file(self, epoch, total_episodes, total_reward):
        out = "{},{},{},{}\n".format(epoch, total_episodes, total_reward, total_reward/total_episodes)
        self.results_file.write(out)
        self.results_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.episode_reward = 0

        self.trace_list.trace_list = []
        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _choose_action(self, trace_list, qec_table, epsilon, observation, reward):
        trace_list.add_trace(self.last_img, self.last_action, reward, False)

        # epsilon greedy
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)

        value = -100
        maximum_action = 0
        # argmax(Q(s,a))
        for action in range(self.num_actions):
            value_t = qec_table.estimate(observation, action)
            if value_t > value:
                value = value_t
                maximum_action = action

        return maximum_action

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1
        self.episode_reward += reward

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)

        action = self._choose_action(self.trace_list, self.qec_table, self.epsilon, observation, np.clip(reward, -1, 1))

        self.last_action = action
        self.last_img = observation

        return action

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.total_reward += self.episode_reward
        self.total_episodes += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        # Store the latest sample.
        self.trace_list.add_trace(self.last_img, self.last_action, np.clip(reward, -1, 1), True)
        """
        do update
        """
        q_return = 0.
        for i in range(len(self.trace_list.trace_list)-1, -1, -1):
            node = self.trace_list.trace_list[i]
            q_return = q_return * self.ec_discount + node.reward
            self.qec_table.update(node.image, node.action, q_return)

        # calculate time
        rho = 0.98
        self.steps_sec_ema *= rho
        self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)
        logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
            self.step_counter/total_time, self.steps_sec_ema))
        logging.info('episode {} reward: {:.2f}'.format(self.total_episodes, self.episode_reward))

    def finish_epoch(self, epoch):
        qec_file = open(self.exp_dir + '/qec_table_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.qec_table, qec_file, 2)
        qec_file.close()

        self._update_results_file(epoch, self.total_episodes, self.total_reward)
        self.total_episodes = 0
        self.total_reward = 0

        # EC_functions.print_table(self.qec_table)

