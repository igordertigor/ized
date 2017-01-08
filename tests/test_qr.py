import unittest

import numpy as np

from ized import qr


class QRchecker(unittest.TestCase):

    def setUp(self):
        self.test_matrix = np.r_[
            np.eye(4) - np.eye(4, k=1) - np.eye(4, k=-1),
            np.ones((1, 4))]

    def assertUpperTriangular(self, mat):
        nrows = mat.shape[0]
        self.assertSquareMatrix(mat)
        for i in range(nrows):
            for j in range(i+1, nrows):
                self.assertEqual(mat[j, i], 0)

    def assertSquareMatrix(self, mat):
        nrows = mat.shape[0]
        self.assertEqual(mat.shape, (nrows, nrows))


class TestQRmapped(QRchecker):

    def test_qr_mapped_returns_upper_triangular_of_large_matrix(self):
        upper_triangular = qr.qr_mapped(self.test_matrix, n=4)
        self.assertUpperTriangular(upper_triangular)

    def test_qr_mapped_just_returns_the_same_for_small_matrix(self):
        not_upper_triangular = qr.qr_mapped(self.test_matrix, n=6)

        self.assertEqual(not_upper_triangular.shape, self.test_matrix.shape)

        self.assertTrue(np.all((not_upper_triangular - self.test_matrix) == 0))


class TestQRreduce(QRchecker):

    def test_qr_reduce_reduces_to_upper_triangular_if_matrices_are_big(self):
        upper_triangular = qr.qr_reduce(self.test_matrix,
                                        self.test_matrix,
                                        n=4)
        self.assertUpperTriangular(upper_triangular)

    def test_qr_reduce_does_nothing_for_small_matricies(self):
        not_upper_triangular = qr.qr_reduce(self.test_matrix,
                                            self.test_matrix,
                                            n=11)
        self.assertEqual(not_upper_triangular.shape, (10, 4))


class TestMapReduceQr(QRchecker):

    def test_on_iterator(self):
        x_chunks = (self.test_matrix for _ in range(5))
        upper_triangular = qr.mapreduce_qr(x_chunks)
        self.assertUpperTriangular(upper_triangular)

    def test_lm_solve_qr(self):
        x_chunks = (self.test_matrix for _ in range(5))
        w, r2, _ = qr.lm_solve_qr(x_chunks)

        self.assertGreater(r2, 0)
        self.assertEqual(len(w.shape), 1)
        self.assertEqual(w.shape, (3,))
