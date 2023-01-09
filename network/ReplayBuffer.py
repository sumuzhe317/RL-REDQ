import numpy as np
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, buf_size):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        """
        ## init buffers as numpy arrays
        self.obs_buf = np.zeros([buf_size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([buf_size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([buf_size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buf_size

    def store(self, obs, act, rew, next_obs, done):
        """
        data will get stored in the pointer's location
        """
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        ## move the pointer to store in next location in buffer
        self.ptr = (self.ptr+1) % self.max_size
        ## keep track of the current buffer size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs_buf[idxs],
                    obs2=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    idxs=idxs)