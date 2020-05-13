import numpy as np
import scipy.stats as st
from scipy.stats import chi2

def find_tux_mean(interval, sample):
    return st.t.interval(interval, len(sample)-1, loc=np.mean(sample), scale=st.sem(sample))[1]

def take_sample(mu=10, sigma=4, n=1000):
    return np.random.normal(mu, sigma, n)

def find_tux_std(interval, sample):
    return (((len(sample) - 1) * (np.std(sample, ddof =1) **2) )/chi2.ppf((1 - interval)/2, len(sample) - 1)) ** 0.5

def find_tux(tux_mean, tux_std):
    return tux_mean + (2 * tux_std)




class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.id = id # integer
        self.group = group
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.eval_data = {k: np.array(v) for k, v in eval_data.items()}
        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.eval_data['y'])
        self.eval_samples = len(self.eval_data['y'])

        self.total_samples = self.num_samples + self.test_samples + self.eval_samples

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))
    
    def find_total_time(self, batch_size, mu=10, sigma=4):
        total_batches = int(self.total_samples/batch_size)
        return np.sum(np.random.normal(mu, sigma, total_batches))

    def solve_inner(self, num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''
        total_time = self.find_total_time(batch_size)
        sample = take_sample()
        tux = find_tux(find_tux_mean(0.95, sample), find_tux_std(0.95, sample))

        bytes_w = self.model.size
        soln, comp = self.model.solve_inner(self.train_data, num_epochs, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r),  int(self.total_samples/batch_size), tux, total_time

    def solve_iters(self, num_iters=1, batch_size=10):
        '''Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size
        soln, comp = self.model.solve_iters(self.train_data, num_iters, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.num_samples


    def test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss = self.model.test(self.eval_data)
        return tot_correct, self.test_samples
