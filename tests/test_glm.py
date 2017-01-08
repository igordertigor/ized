import unittest

import numpy as np

from ized import glm


def logistic(x):
    return 1./(1 + np.exp(-x))


def ilogistic(y):
    return np.log(y/(1-y))


def softrelu(x):
    return np.log1p(np.exp(x))


def isoftrelu(y):
    return np.log(np.exp(y)-1)


def numerical_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x))/h


class TestLink(unittest.TestCase):

    def setUp(self):
        self.eta = np.array([-1., 0., 1.])

    def test_normal_identity_family(self):
        y = np.zeros(3, 'd')
        info = glm.normal_identity_family(self.eta, y)
        self.check_info(info, y, self.eta, 1., 1., 1.)

    def check_info(self, info, y, mu, e_dmudeta, e_detadmu, e_vary):
        self.assertEqual(info.y.tolist(), y.tolist())
        self.assertEqual(info.mu.tolist(), mu.tolist())

        # Relax equality for derivatives
        self.assertAlmostEqualArrays(info.dmudeta,  e_dmudeta, 1e-4)
        self.assertAlmostEqualArrays(info.detadmu,  e_detadmu, 1e-4)

        self.assertAlmostEqualArrays(info.vary, e_vary)

    def assertAlmostEqualArrays(self, a1, a2, e=1e-7):
        difference = np.max(abs(a1 - a2))
        self.assertTrue(difference < e,
                        'Arrays differ by up to {}\n  a1={}\n  a2={}'.format(
                            difference,
                            a1,
                            a2
                        ))

    def test_binomial_logistic(self):
        y = np.array([0., 1., 0.])
        info = glm.binomial_logistic_family(self.eta, y)
        e_dmudeta = numerical_derivative(logistic, self.eta)
        e_detadmu = numerical_derivative(ilogistic, logistic(self.eta))
        vary = logistic(self.eta)*(1-logistic(self.eta))
        self.check_info(
            info, y, logistic(self.eta), e_dmudeta, e_detadmu, vary)

    def test_poisson_log(self):
        y = np.zeros(3, 'd')
        info = glm.poisson_log_family(self.eta, y)
        e_dmudeta = numerical_derivative(np.exp, self.eta)
        e_detadmu = numerical_derivative(np.log, np.exp(self.eta))
        vary = np.exp(self.eta)
        self.check_info(info, y, np.exp(self.eta), e_dmudeta, e_detadmu, vary)

    def test_poisson_softrelu(self):
        y = np.zeros(3, 'd')
        info = glm.poisson_softrelu_family(self.eta, y)
        mu = softrelu(self.eta)
        e_dmudeta = numerical_derivative(softrelu, self.eta)
        e_detadmu = numerical_derivative(isoftrelu, mu)
        vary = mu
        self.check_info(info, y, mu, e_dmudeta, e_detadmu, vary)

        self.assertAlmostEqualArrays(self.eta, isoftrelu(mu))
