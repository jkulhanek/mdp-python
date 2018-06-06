# -*- coding: UTF-8 -*-

from ZUI_MDP_solution import *
from unittest import TestCase
import itertools as it
import numpy as np
import numpy.testing as nptest


# Taken from http://www.neuraldump.net/2017/06/how-to-suppress-python-unittest-warnings/.
def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_func(self, *args, **kwargs)
    return do_test


class TestGridWorld2x2(TestCase):
    rtol = 1e-4 # relative tolerance for comparing two floats

    def setUp(self):
        self.gw = GridWorld.get_world('2x2')

    def test_is_obstacle_at(self):
        self.assertFalse(self.gw._is_obstacle([0, 0]))
        self.assertFalse(self.gw._is_obstacle([0, 1]))
        self.assertFalse(self.gw._is_obstacle([1, 0]))
        self.assertFalse(self.gw._is_obstacle([1, 1]))

    def test_is_on_grid_true(self):
        self.assertTrue(self.gw._is_on_grid([0, 0]),msg='The point [{},{}] should be on the grid.'.format(0, 0))
        self.assertTrue(self.gw._is_on_grid([1, 1]), msg='The point [{},{}] should be on the grid.'.format(1, 1))

    def test_is_on_grid_false(self):
        for point in ([-1,0], [-2,-2], [2,0], [0,2], [5,5], [0,-1]):
             self.assertFalse(self.gw._is_on_grid(point),msg='The point [{}] should not be on the grid.'.format(point))

    def test_transition_proba(self):
        true_transition_proba = np.load('./test_data/test_gw_2x2_transition_proba.npy')
        nptest.assert_allclose(self.gw.transition_proba,true_transition_proba,rtol=self.rtol)

    def test_Q_from_V_zeros(self):
        V = np.zeros((self.gw.n_states + 1,))
        desired_Q = np.array([[-0.04, -0.04, -0.04, -0.04],
                              [1., 1., 1., 1.],
                              [-0.04, -0.04, -0.04, -0.04],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])
        nptest.assert_allclose(self.gw.Q_from_V(V=V), desired_Q, rtol=self.rtol)

    def test_Q_from_V_ones(self):
        V = np.ones((self.gw.n_states + 1,))
        desired_Q = np.array([[0.96, 0.96, 0.96, 0.96],
                              [2., 2., 2., 2.],
                              [0.96, 0.96, 0.96, 0.96],
                              [0., 0., 0., 0.],
                              [1., 1., 1., 1.]])
        nptest.assert_allclose(self.gw.Q_from_V(V=V), desired_Q, rtol=self.rtol)

    def test_Q_from_V_init(self):
        V = np.max(self.gw.rewards,axis=1)
        desired_Q = np.array([[0.024, 0.752, 0.024, -0.08],
                              [1., 1., 1., 1.],
                              [-0.176, -0.848, -0.176, -0.08],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])
        nptest.assert_allclose(self.gw.Q_from_V(V=V), desired_Q, rtol=self.rtol)
    def test_Q2V_single(self):
        desired_V = np.array([0.752, 1., -0.08, -1., 0.])
        Q = np.array([[0.024, 0.752, 0.024, -0.08],
                      [1., 1., 1., 1.],
                      [-0.176, -0.848, -0.176, -0.08],
                      [-1., -1., -1., -1.],
                      [0., 0., 0., 0.]])

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2V(self):
        desired_V = np.array([0.752, 1., -0.08, -1., 0.])
        Q = np.array([[0.024, 0.752, 0.024, -0.08],
                      [1., 1., 1., 1.],
                      [-0.176, -0.848, -0.176, -0.08],
                      [-1., -1., -1., -1.],
                      [0., 0., 0., 0.]])

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2Vbypolicy(self):
        desired_V = np.array([0.9178081, 1., 0.66027364, -1., 0., ])
        Q = np.array([[0.88602712, 0.9178081, 0.67999927, 0.85205443],
                              [1., 1., 1., 1.],
                              [0.66027364, -0.6821919, 0.45424578, 0.64602658],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])
        policy = np.array([1, 0, 0, 0, 0], dtype=int)
        actual_V = self.gw.Q2Vbypolicy(Q,policy)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2Vbypolicy should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2policy(self):
        Q = np.array([[0.88602712, 0.9178081, 0.67999927, 0.85205443],
                              [1., 1., 1., 1.],
                              [0.66027364, -0.6821919, 0.45424578, 0.64602658],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])
        desired_policy = np.array([1, 0, 0, 0, 0], dtype=int)
        actual_policy = self.gw.Q2policy(Q)
        self.assertEqual(actual_policy.shape, (self.gw.n_states + 1,), msg='Q2policy should return array policy of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_policy.shape))
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    @ignore_warnings
    def test_value_iteration_single_iter(self):
        actual_Q = self.gw.value_iteration(max_iter=1)
        desired_Q = np.array([[0.024, 0.752, 0.024, -0.08],
                              [1., 1., 1., 1.],
                              [-0.176, -0.848, -0.176, -0.08],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])

        desired_Q_shape = (self.gw.n_states + 1,self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Value_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                                                                                  desired_Q_shape,
                                                                                  actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)

    def test_value_iteration(self):
        desired_Q = np.array([[0.88602712, 0.9178081, 0.67999927, 0.85205443],
                              [1., 1., 1., 1.],
                              [0.66027364, -0.6821919, 0.45424578, 0.64602658],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])
        actual_Q = self.gw.value_iteration()
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)

    def test_policy_iteration_policy_only(self):
        actual_policy = self.gw.Q2policy(self.gw.policy_iteration())
        desired_Q = np.array([[0.88602712, 0.9178081, 0.67999927, 0.85205443],
                              [1., 1., 1., 1.],
                              [0.66027364, -0.6821919, 0.45424578, 0.64602658],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])
        desired_policy = self.gw.Q2policy(desired_Q)
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    def test_policy_iteration(self):
        actual_Q = self.gw.policy_iteration()
        desired_Q = np.array([[0.88602712, 0.9178081, 0.67999927, 0.85205443],
                              [1., 1., 1., 1.],
                              [0.66027364, -0.6821919, 0.45424578, 0.64602658],
                              [-1., -1., -1., -1.],
                              [0., 0., 0., 0.]])
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Policy_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                             desired_Q_shape,
                             actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)


class TestGridWorld3x3(TestCase):
    rtol = 1e-3 # relative tolerance for comparing two floats

    def setUp(self):
        self.gw = GridWorld.get_world('3x3')

    def test_is_obstacle_at(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            self.assertFalse(self.gw._is_obstacle([i, j]), msg='No obstacle should be at [{},{}].'.format(i,j))

    def test_is_on_grid_true(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            self.assertTrue(self.gw._is_on_grid([i, j]),msg='The point [{},{}] should be on the grid.'.format(i, j))

    def test_is_on_grid_false(self):
        for point in ([-1,0], [-2,-2], [3,0], [0,3], [5,5], [0,-1]):
             self.assertFalse(self.gw._is_on_grid(point),msg='The point [{}] should not be on the grid.'.format(point))

    def test_transition_proba(self):
        true_transition_proba = np.load('./test_data/test_gw_3x3_transition_proba.npy')
        nptest.assert_allclose(self.gw.transition_proba,true_transition_proba,rtol=self.rtol)

    def test_Q2V_single(self):
        desired_V = np.load('./test_data/test_gw_3x3_V_single_iter.npy')
        Q = np.load('./test_data/test_gw_3x3_Q_single_iter.npy')

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2V(self):
        desired_V = np.load('./test_data/test_gw_3x3_V.npy')
        Q = np.load('./test_data/test_gw_3x3_Q.npy')
        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2Vbypolicy(self):
        desired_V = np.load('./test_data/test_gw_3x3_V.npy')
        Q = np.load('./test_data/test_gw_3x3_Q.npy')
        policy = np.array([1, 1, 0, 0, 3, 0, 0, 3, 2, 0],dtype=int)
        actual_V = self.gw.Q2Vbypolicy(Q, policy)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2Vbypolicy should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2policy(self):
        Q = np.load('./test_data/test_gw_3x3_Q.npy')
        desired_policy = np.array([1, 1, 0, 0, 3, 0, 0, 3, 2, 0],dtype=int)
        actual_policy = self.gw.Q2policy(Q)
        self.assertEqual(actual_policy.shape, (self.gw.n_states + 1,), msg='Q2policy should return array policy of'
                                                                           ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_policy.shape))
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    @ignore_warnings
    def test_value_iteration_single_iter(self):
        actual_Q = self.gw.value_iteration(max_iter=1)
        desired_Q = np.load('./test_data/test_gw_3x3_Q_single_iter.npy')
        desired_Q_shape = (self.gw.n_states + 1,self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Value_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                                                                                  desired_Q_shape,
                                                                                  actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)

    def test_value_iteration(self):
        desired_Q = np.load('./test_data/test_gw_3x3_Q.npy')
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        actual_Q = self.gw.value_iteration()
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)

    def test_policy_iteration_policy_only(self):
        actual_policy = self.gw.Q2policy(self.gw.policy_iteration())
        desired_Q = np.load('./test_data/test_gw_3x3_Q.npy')
        desired_policy = self.gw.Q2policy(desired_Q)
        #actual_policy = self.gw.Q2policy(actual_Q)
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    def test_policy_iteration(self):
        actual_Q = self.gw.policy_iteration()
        desired_Q = np.load('./test_data/test_gw_3x3_Q.npy')
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Policy_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                             desired_Q_shape,
                             actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)


class TestGridWorld3x4(TestCase):
    rtol = 1e-3 # relative tolerance for comparing two floats

    def setUp(self):
        self.gw = GridWorld.get_world('3x4')

    def test_is_obstacle_at(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            if i == 1 and j == 1:
                continue
            self.assertFalse(self.gw._is_obstacle([i, j]), msg='No obstacle should be at [{},{}].'.format(i,j))
        self.assertTrue(self.gw._is_obstacle([1, 1]), msg='An obstacle should be at [{},{}].'.format(1, 1))

    def test_is_on_grid_true(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            self.assertTrue(self.gw._is_on_grid([i, j]),msg='The point [{},{}] should be on the grid.'.format(i, j))

    def test_is_on_grid_false(self):
        for point in ([-1,0], [-2,-2], [3,0], [0,4], [5,5], [0,-1]):
             self.assertFalse(self.gw._is_on_grid(point),msg='The point [{}] should not be on the grid.'.format(point))

    def test_transition_proba(self):
        true_transition_proba = np.load('./test_data/test_gw_3x4_transition_proba.npy')
        nptest.assert_allclose(self.gw.transition_proba,true_transition_proba,rtol=self.rtol)

    def test_Q2V_single(self):
        desired_V = np.load('./test_data/test_gw_3x4_V_single_iter.npy')
        Q = np.load('./test_data/test_gw_3x4_Q_single_iter.npy')

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2V(self):
        desired_V = np.load('./test_data/test_gw_3x4_V.npy')
        Q = np.load('./test_data/test_gw_3x4_Q.npy')

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    def test_Q2policy(self):
        Q = np.load('./test_data/test_gw_3x4_Q.npy')
        desired_policy = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],dtype=int)
        actual_policy = self.gw.Q2policy(Q)
        self.assertEqual(actual_policy.shape, (self.gw.n_states + 1,), msg='Q2policy should return array policy of'
                                                                           ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_policy.shape))
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    def test_Q2Vbypolicy(self):
        desired_V = np.load('./test_data/test_gw_3x4_V.npy')
        Q = np.load('./test_data/test_gw_3x4_Q.npy')
        policy = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0],dtype=int)
        actual_V = self.gw.Q2Vbypolicy(Q, policy)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2Vbypolicy should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol)

    @ignore_warnings
    def test_value_iteration_single_iter(self):
        actual_Q = self.gw.value_iteration(max_iter=1)
        desired_Q = np.load('./test_data/test_gw_3x4_Q_single_iter.npy')
        desired_Q_shape = (self.gw.n_states + 1,self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Value_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                                                                                  desired_Q_shape,
                                                                                  actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)

    def test_value_iteration(self):
        desired_Q = np.load('./test_data/test_gw_3x4_Q.npy')
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        actual_Q = self.gw.value_iteration()
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)

    def test_policy_iteration_policy_only(self):
        actual_policy = self.gw.Q2policy(self.gw.policy_iteration())
        desired_Q = np.load('./test_data/test_gw_3x4_Q.npy')
        desired_policy = self.gw.Q2policy(desired_Q)
        #actual_policy = self.gw.Q2policy(actual_Q)
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    def test_policy_iteration(self):
        actual_Q = self.gw.policy_iteration()
        desired_Q = np.load('./test_data/test_gw_3x4_Q.npy')
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Policy_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                             desired_Q_shape,
                             actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol)


class TestGridWorld4x4(TestCase):
    rtol = 1e-3  # relative tolerance for comparing two floats
    atol = 1e-08  # absolute tolerance for comparing two floats

    def setUp(self):
        self.gw = GridWorld.get_world('4x4')

    def test_is_obstacle_at(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            if (i,j) in [(1,1),(2,2)]:
                continue
            self.assertFalse(self.gw._is_obstacle([i, j]), msg='No obstacle should be at [{},{}].'.format(i,j))
        self.assertTrue(self.gw._is_obstacle([1, 1]), msg='An obstacle should be at [{},{}].'.format(1, 1))

    def test_is_on_grid_true(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            self.assertTrue(self.gw._is_on_grid([i, j]),msg='The point [{},{}] should be on the grid.'.format(i, j))

    def test_is_on_grid_false(self):
        for point in ([-1,0], [-2,-2], [4,0], [0,4], [5,5], [0,-1]):
             self.assertFalse(self.gw._is_on_grid(point),msg='The point [{}] should not be on the grid.'.format(point))

    def test_transition_proba(self):
        true_transition_proba = np.load('./test_data/test_gw_4x4_transition_proba.npy')
        nptest.assert_allclose(self.gw.transition_proba,true_transition_proba,rtol=self.rtol, atol=self.atol)

    def test_Q2V_single(self):
        desired_V = np.load('./test_data/test_gw_4x4_V_single_iter.npy')
        Q = np.load('./test_data/test_gw_4x4_Q_single_iter.npy')

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol, atol=self.atol)

    def test_Q2V(self):
        desired_V = np.load('./test_data/test_gw_4x4_V.npy')
        Q = np.load('./test_data/test_gw_4x4_Q.npy')

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol, atol=self.atol)

    def test_Q2Vbypolicy(self):
        desired_V = np.load('./test_data/test_gw_4x4_V.npy')
        Q = np.load('./test_data/test_gw_4x4_Q.npy')
        policy = np.load('./test_data/test_gw_4x4_policy.npy')
        actual_V = self.gw.Q2Vbypolicy(Q, policy)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2Vbypolicy should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol, atol=self.atol)

    def test_Q2policy(self):
        Q = np.load('./test_data/test_gw_4x4_Q.npy')
        desired_policy = np.load('./test_data/test_gw_4x4_policy.npy')
        actual_policy = self.gw.Q2policy(Q)
        self.assertEqual(actual_policy.shape, (self.gw.n_states + 1,), msg='Q2policy should return array policy of'
                                                                           ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_policy.shape))
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    @ignore_warnings
    def test_value_iteration_single_iter(self):
        actual_Q = self.gw.value_iteration(max_iter=1)
        desired_Q = np.load('./test_data/test_gw_4x4_Q_single_iter.npy')
        desired_Q_shape = (self.gw.n_states + 1,self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Value_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                                                                                  desired_Q_shape,
                                                                                  actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol, atol=self.atol)

    def test_value_iteration(self):
        desired_Q = np.load('./test_data/test_gw_4x4_Q.npy')
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        actual_Q = self.gw.value_iteration()
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol, atol=self.atol)

    def test_policy_iteration_policy_only(self):
        actual_policy = self.gw.Q2policy(self.gw.policy_iteration())
        desired_Q = np.load('./test_data/test_gw_4x4_Q.npy')
        desired_policy = self.gw.Q2policy(desired_Q)
        #actual_policy = self.gw.Q2policy(actual_Q)
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol, atol=self.atol)

    def test_policy_iteration(self):
        actual_Q = self.gw.policy_iteration()
        desired_Q = np.load('./test_data/test_gw_4x4_Q.npy')
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Policy_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                             desired_Q_shape,
                             actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol, atol=self.atol)


class TestGridWorld5x5(TestCase):
    rtol = 1e-4  # relative tolerance for comparing two floats
    atol = 1e-08  # absolute tolerance for comparing two floats

    def setUp(self):
        self.gw = GridWorld.get_world('5x5')

    def test_is_obstacle_at(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            if (i,j) in [(1,0), (1,1), (2,2)]:
                continue
            self.assertFalse(self.gw._is_obstacle([i, j]), msg='No obstacle should be at [{},{}].'.format(i,j))
        self.assertTrue(self.gw._is_obstacle([1, 1]), msg='An obstacle should be at [{},{}].'.format(1, 1))

    def test_is_on_grid_true(self):
        for i,j in it.product(range(self.gw.n_rows),range(self.gw.n_columns)):
            self.assertTrue(self.gw._is_on_grid([i, j]),msg='The point [{},{}] should be on the grid.'.format(i, j))

    def test_is_on_grid_false(self):
        for point in ([-1,0], [-2,-2], [5,0], [0,5], [5,5], [0,-1]):
             self.assertFalse(self.gw._is_on_grid(point),msg='The point [{}] should not be on the grid.'.format(point))

    def test_transition_proba(self):
        true_transition_proba = np.load('./test_data/test_gw_5x5_transition_proba.npy')
        nptest.assert_allclose(self.gw.transition_proba,true_transition_proba,rtol=self.rtol, atol=self.atol)

    def test_Q2V_single(self):
        desired_V = np.load('./test_data/test_gw_5x5_V_single_iter.npy')
        Q = np.load('./test_data/test_gw_5x5_Q_single_iter.npy')

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol, atol=self.atol)

    def test_Q2V(self):
        desired_V = np.load('./test_data/test_gw_5x5_V.npy')
        Q = np.load('./test_data/test_gw_5x5_Q.npy')

        actual_V = self.gw.Q2V(Q)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2V should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        actual_V = self.gw.Q2V(Q)
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol, atol=self.atol)

    def test_Q2Vbypolicy(self):
        desired_V = np.load('./test_data/test_gw_5x5_V.npy')
        Q = np.load('./test_data/test_gw_5x5_Q.npy')
        policy = np.load('./test_data/test_gw_5x5_policy.npy')
        actual_V = self.gw.Q2Vbypolicy(Q, policy)
        self.assertEqual(actual_V.shape, (self.gw.n_states + 1,), msg='Q2Vbypolicy should return array V of'
                                                                      ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_V.shape))
        nptest.assert_allclose(actual_V, desired_V, rtol=self.rtol, atol=self.atol)

    def test_Q2policy(self):
        Q = np.load('./test_data/test_gw_5x5_Q.npy')
        desired_policy = np.load('./test_data/test_gw_5x5_policy.npy')
        actual_policy = self.gw.Q2policy(Q)
        self.assertEqual(actual_policy.shape, (self.gw.n_states + 1,), msg='Q2policy should return array policy of'
                                                                           ' shape {} but has returned V with shape {}.'.format(
            self.gw.n_states + 1,
            actual_policy.shape))
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol)

    @ignore_warnings
    def test_value_iteration_single_iter(self):
        actual_Q = self.gw.value_iteration(max_iter=1)
        desired_Q = np.load('./test_data/test_gw_5x5_Q_single_iter.npy')
        desired_Q_shape = (self.gw.n_states + 1,self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Value_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                                                                                  desired_Q_shape,
                                                                                  actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol, atol=self.atol)

    def test_value_iteration(self):
        desired_Q = np.load('./test_data/test_gw_5x5_Q.npy')
        actual_Q = self.gw.value_iteration()
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol, atol=self.atol)

    def test_policy_iteration_policy_only(self):
        actual_policy = self.gw.Q2policy(self.gw.policy_iteration())
        desired_Q = np.load('./test_data/test_gw_5x5_Q.npy')
        desired_policy = self.gw.Q2policy(desired_Q)
        nptest.assert_allclose(actual_policy, desired_policy, rtol=self.rtol, atol=self.atol)

    def test_policy_iteration(self):
        actual_Q = self.gw.policy_iteration()
        desired_Q = np.load('./test_data/test_gw_5x5_Q.npy')
        desired_Q_shape = (self.gw.n_states + 1, self.gw.n_actions)
        self.assertEqual(actual_Q.shape, desired_Q_shape,
                         msg='Policy_iteration should return array Q of'
                             ' shape {} but has returned V with shape {}.'.format(
                             desired_Q_shape,
                             actual_Q.shape))
        nptest.assert_allclose(actual_Q, desired_Q, rtol=self.rtol, atol=self.atol)