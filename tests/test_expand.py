import unittest

import numpy as np

from ized import expand


class TestBsplines(unittest.TestCase):

    def test_evaluate_bspline_of_degree_0(self):
        x = np.array([-0.5, 0.5, 1.5])
        knots = np.array([0., 1., 2.])
        fvals = expand.bspline(x, knots, 0, 0)

        self.assertEqual(fvals.shape, x.shape)
        self.assertEqual(fvals[0], 0)
        self.assertEqual(fvals[1], 1)
        self.assertEqual(fvals[2], 0)

    def test_evaluate_bspline_of_degree_1(self):
        x = np.array([0, 0.5, 1., 1.5, 2.])
        knots = np.array([0., 0., 1., 2., 2.])
        fvals = expand.bspline(x, knots, 1, 1)

        self.assertEqual(fvals.shape, x.shape)
        self.assertEqual(fvals.tolist(), [0, .5, 1., .5, 0.])

    def test_evaluate_bspline_of_degree_2(self):
        x = np.array([0, 0.5, 1., 1.5, 2., 2.5, 3.])
        knots = np.array([0., 0., 0., 1., 2., 3., 3., 3.])
        fvals = expand.bspline(x, knots, 2, 2)

        self.assertEqual(fvals.shape, x.shape)
        self.assertEqual(fvals.tolist(), [0, .125, .5, .75, .5, .125, 0])


class TestBsplineExpansion(unittest.TestCase):

    def test_expansion_gives_right_shape(self):
        x = np.array([0, 0.5, 1., 1.5, 2., 2.5, 3.])
        knots = np.array([0., 0., 0., 1., 2., 3., 3., 3.])
        expanded = expand.bs_expand(x, knots, 2)
        self.assertEqual(expanded.shape, (len(x), len(knots) + 1))

        # Values are tested already


class TestBsplinePenalty(unittest.TestCase):

    def test_shape_of_penalty_agrees_with_number_of_basis_functions(self):
        x = np.array([0, 0.5, 1., 1.5, 2., 2.5, 3.])
        knots = np.array([0., 0., 0., 1., 2., 3., 3., 3.])
        penalty = expand.bs_penalty(knots, 2)
        expanded = expand.bs_expand(x, knots, 2)

        self.assertEqual(penalty.shape[0], penalty.shape[1])
        self.assertEqual(penalty.shape[1], expanded.shape[1])

    def test_penalty_is_triband(self):
        knots = np.array([0., 0., 0., 1., 2., 3., 3., 3.])
        penalty = expand.bs_penalty(knots, 2)
        nzidx = np.where(penalty)
        for x, y in zip(*nzidx):
            self.assertIn(y, {x, x+1, x+2})
