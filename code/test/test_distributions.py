import argparse
import torch
import unittest
import cdae.distributions as dists
import cdae.util as util

# Fake parsed arguments
opt = argparse.Namespace(
    seed=3,
    device=-1,
    cuda=False,
    visdom=True
)
util.init(opt)


class TestCategorical(unittest.TestCase):
    def test_dimensions(self):
        # probabilities: Tensor [num_categories]
        # categories: Tensor [num_categories]
        # value: Tensor [1]
        # logpdf: Tensor [1]
        uniforms = torch.rand(3)
        probabilities = uniforms / torch.sum(uniforms)
        categories = torch.rand(3)
        value = dists.categorical_sample(categories, probabilities)
        self.assertEqual(
            value.size(),
            torch.Size([1])
        )
        logpdf = dists.categorical_logpdf(value, categories, probabilities)
        self.assertEqual(
            logpdf.size(),
            torch.Size([1])
        )

        # probabilities: Tensor [num_categories, dim_1, ..., dim_N]
        # categories: Tensor [num_categories, dim_1, ..., dim_N]
        # value: Tensor [dim_1, ..., dim_N]
        # logpdf: Tensor [dim_1, ..., dim_N]
        uniforms = torch.rand(3, 4, 5)
        probabilities = uniforms / torch.sum(uniforms, dim=0).expand_as(uniforms)
        categories = torch.rand(3, 4, 5)
        value = dists.categorical_sample(categories, probabilities)
        self.assertEqual(
            value.size(),
            torch.Size([4, 5])
        )
        logpdf = dists.categorical_logpdf(value, categories, probabilities)
        self.assertEqual(
            logpdf.size(),
            torch.Size([4, 5])
        )

if __name__ == '__main__':
    unittest.main()
