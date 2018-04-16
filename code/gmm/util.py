import torch
import numpy as np

WORKING_DIR = './working_files'
OUTPUT_DIR = './output_files'


class OnlineMeanStd():
    def __init__(self):
        self.count = 0
        self.means = None
        self.M2s = None

    def update(self, new_variables):
        if self.count == 0:
            self.count = 1
            self.means = []
            self.M2s = []
            for new_var in new_variables:
                self.means.append(new_var.data)
                self.M2s.append(new_var.data.new(new_var.size()).fill_(0))
        else:
            self.count = self.count + 1
            for new_var_idx, new_var in enumerate(new_variables):
                delta = new_var.data - self.means[new_var_idx]
                self.means[new_var_idx] = self.means[new_var_idx] + delta / self.count
                delta_2 = new_var.data - self.means[new_var_idx]
                self.M2s[new_var_idx] = self.M2s[new_var_idx] + delta * delta_2

    def means_stds(self):
        if self.count < 2:
            raise ArithmeticError('Need more than 1 value. Have {}'.format(self.count))
        else:
            stds = []
            for i in range(len(self.means)):
                stds.append(torch.sqrt(self.M2s[i] / self.count))
            return self.means, stds

    def avg_of_means_stds(self):
        means, stds = self.means_stds()
        num_parameters = np.sum([len(p) for p in means])
        return (
            np.sum([torch.sum(p) for p in means]) / num_parameters,
            np.sum([torch.sum(p) for p in stds]) / num_parameters
        )
