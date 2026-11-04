"""
GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved

See LICENSE for the license information.

Unit tests for wrapped Marginals methods added for covariance recovery.
Author: Codex, prompted by Frank Dellaert
"""

# pylint: disable=invalid-name, no-name-in-module, no-member

import unittest

import numpy as np

import gtsam
from gtsam.symbol_shorthand import L, X
from gtsam.utils.test_case import GtsamTestCase


def create_planar_slam_problem():
    """Create a small nonlinear SLAM problem with pose and landmark variables."""
    graph = gtsam.NonlinearFactorGraph()

    x1 = X(1)
    x2 = X(2)
    x3 = X(3)
    l1 = L(1)
    l2 = L(2)

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))

    graph.addPriorPose2(x1, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise)
    graph.add(
        gtsam.BetweenFactorPose2(x1, x2, gtsam.Pose2(2.0, 0.0, 0.0), odometry_noise)
    )
    graph.add(
        gtsam.BetweenFactorPose2(x2, x3, gtsam.Pose2(2.0, 0.0, 0.0), odometry_noise)
    )

    graph.add(
        gtsam.BearingRangeFactor2D(
            x1, l1, gtsam.Rot2.fromDegrees(45), np.sqrt(8.0), measurement_noise
        )
    )
    graph.add(
        gtsam.BearingRangeFactor2D(
            x2, l1, gtsam.Rot2.fromDegrees(90), 2.0, measurement_noise
        )
    )
    graph.add(
        gtsam.BearingRangeFactor2D(
            x3, l2, gtsam.Rot2.fromDegrees(90), 2.0, measurement_noise
        )
    )

    values = gtsam.Values()
    values.insert(x1, gtsam.Pose2(0.0, 0.0, 0.0))
    values.insert(x2, gtsam.Pose2(2.0, 0.0, 0.0))
    values.insert(x3, gtsam.Pose2(4.0, 0.0, 0.0))
    values.insert(l1, gtsam.Point2(2.0, 2.0))
    values.insert(l2, gtsam.Point2(4.0, 2.0))

    return graph, values, (x1, x2, x3, l1, l2)


def create_wnoa_problem(dt, qc=1.0):
    """Create a linear 1-D constant-velocity GP smoother with two states.

    Each state is [position, velocity]. Both positions are measured with a
    0.5 m sigma, the first velocity has a 10 m/s prior sigma, and the
    white-noise-on-acceleration motion factor uses
    Q(dt) = qc * [[dt^3/3, dt^2/2], [dt^2/2, dt]].
    For small dt this full-rank system has a tiny normalized Cholesky pivot.
    """
    x0, x1 = X(0), X(1)
    position = np.array([[1.0, 0.0]])
    velocity = np.array([[0.0, 1.0]])
    transition = np.array([[1.0, dt], [0.0, 1.0]])
    process_covariance = qc * np.array(
        [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
    )

    graph = gtsam.GaussianFactorGraph()
    position_noise = gtsam.noiseModel.Isotropic.Sigma(1, 0.5)
    velocity_noise = gtsam.noiseModel.Isotropic.Sigma(1, 10.0)
    graph.add(x0, position, np.zeros(1), position_noise)
    graph.add(x1, position, np.zeros(1), position_noise)
    graph.add(x0, velocity, np.zeros(1), velocity_noise)
    # Wrapped JacobianFactors take diagonal noise, so prewhiten by Q^(-1/2).
    whitening = np.linalg.inv(np.linalg.cholesky(process_covariance))
    graph.add(
        x0, -whitening @ transition, x1, whitening, np.zeros(2),
        gtsam.noiseModel.Unit.Create(2),
    )

    # Dense reference covariance from the information matrix, ordered x0, x1.
    measurement_information = np.zeros((4, 4))
    measurement_information[0, 0] += 1.0 / 0.5**2
    measurement_information[2, 2] += 1.0 / 0.5**2
    measurement_information[1, 1] += 1.0 / 10.0**2
    motion_jacobian = np.hstack([-transition, np.eye(2)])
    information = measurement_information + motion_jacobian.T @ np.linalg.solve(
        process_covariance, motion_jacobian
    )
    solution = gtsam.VectorValues()
    solution.insert(x0, np.zeros(2))
    solution.insert(x1, np.zeros(2))
    return graph, solution, (x0, x1), np.linalg.inv(information)


class TestMarginals(GtsamTestCase):
    """Tests for Marginals wrapper extensions."""

    def test_joint_marginal_blocks(self):
        """JointMarginal blocks should match the full joint covariance."""
        graph, values, (x1, _, x3, l1, l2) = create_planar_slam_problem()
        marginals = gtsam.Marginals(graph, values)

        union_keys = [x1, l1, l2, x3]
        joint = marginals.jointMarginalCovariance(union_keys)
        full = joint.fullMatrix()
        # JointMarginal preserves the order supplied to the query.
        ordered_keys = union_keys
        dims = values.dims()

        offsets = {}
        offset = 0
        for key in ordered_keys:
            offsets[key] = offset
            offset += dims[key]

        def block(key_i, key_j):
            row = offsets[key_i]
            col = offsets[key_j]
            return full[row : row + dims[key_i], col : col + dims[key_j]]

        np.testing.assert_allclose(joint.at(l2, l2), block(l2, l2), atol=1e-8)
        np.testing.assert_allclose(joint.at(x1, l2), block(x1, l2), atol=1e-8)
        np.testing.assert_allclose(joint.at(l1, x3), block(l1, x3), atol=1e-8)
        np.testing.assert_allclose(joint.at(x3, x3), block(x3, x3), atol=1e-8)

    def test_factorization_constructors(self):
        """QR and CHOLESKY constructors, with and without ordering, agree."""
        graph, values, (x1, _, x3, l1, _) = create_planar_slam_problem()
        factorization = gtsam.Marginals.Factorization
        ordering = graph.orderingCOLAMD()
        reference = gtsam.Marginals(graph, values).marginalCovariance(l1)
        for marginals in (
            gtsam.Marginals(graph, values, factorization.CHOLESKY),
            gtsam.Marginals(graph, values, factorization.QR),
            gtsam.Marginals(graph, values, ordering, factorization.QR),
            gtsam.Marginals(graph, values, ordering),
            gtsam.Marginals(graph.linearize(values), values, factorization.QR),
            gtsam.Marginals(
                graph.linearize(values), values, ordering, factorization.QR
            ),
        ):
            np.testing.assert_allclose(
                marginals.marginalCovariance(l1), reference, rtol=1e-9, atol=1e-12
            )
        joint_cholesky = gtsam.Marginals(graph, values).jointMarginalCovariance(
            [x1, x3]
        )
        joint_qr = gtsam.Marginals(
            graph, values, factorization.QR
        ).jointMarginalCovariance([x1, x3])
        np.testing.assert_allclose(
            joint_qr.fullMatrix(), joint_cholesky.fullMatrix(), rtol=1e-9, atol=1e-12
        )

    def test_qr_closely_spaced_wnoa_states(self):
        """QR recovers the dense covariance of closely spaced GP states."""
        factorization = gtsam.Marginals.Factorization
        for dt in (3e-3, 1e-3):
            graph, solution, keys, expected = create_wnoa_problem(dt)
            marginals = gtsam.Marginals(graph, solution, factorization.QR)
            joint = marginals.jointMarginalCovariance(list(keys)).fullMatrix()
            np.testing.assert_allclose(joint, expected, rtol=1e-9)
            np.testing.assert_allclose(
                marginals.marginalCovariance(keys[1]), expected[2:, 2:], rtol=1e-9
            )
            ordering = gtsam.Ordering(list(keys))
            ordered = gtsam.Marginals(graph, solution, ordering, factorization.QR)
            np.testing.assert_allclose(
                ordered.marginalCovariance(keys[0]), expected[:2, :2], rtol=1e-9
            )

    def test_cholesky_rejects_tiny_normalized_pivot(self):
        """CHOLESKY rejects a full-rank WNOA system that QR handles."""
        graph, solution, keys, _ = create_wnoa_problem(1e-3)
        with self.assertRaises(RuntimeError):
            gtsam.Marginals(graph, solution).marginalCovariance(keys[1])


if __name__ == "__main__":
    unittest.main()
