import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h, vx, vy, vw, vh

    contains the bounding box center position (x, y), width w, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, w, h) is taken as a direct observation of the state space (linear
    observation model).
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Initialize motion and update matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard deviation weights for position and velocity.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 0.01

    def initiate(self, measurement):
        """Create track from an unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.

        Returns
        -------
        mean : ndarray
            The 8-dimensional mean vector of the initialized state.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the initialized state.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[2],  # x方向位置不确定性
            self._std_weight_position * mean[3],  # y方向位置不确定性
            self._std_weight_position * mean[2],  # w方向位置不确定性
            self._std_weight_position * mean[3]  # h方向位置不确定性
        ]

        std_vel = [
            self._std_weight_velocity * mean[4],  # vx方向速度不确定性
            self._std_weight_velocity * mean[5],  # vy方向速度不确定性
            self._std_weight_velocity * mean[6],  # vw方向宽度变化率不确定性
            self._std_weight_velocity * mean[7]  # vh方向高度变化率不确定性
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = mean @ self._motion_mat.T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8-dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        mean : ndarray
            The projected mean vector.
        covariance : ndarray
            The projected covariance matrix.
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).

        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrices of the object states at the
            previous time step.

        Returns
        -------
        mean : ndarray
            The predicted mean vectors.
        covariance : ndarray
            The predicted covariance matrices.
        """
        std_pos = self._std_weight_position * mean[:, 2:4]
        std_vel = self._std_weight_velocity * mean[:, 2:4]
        std = np.hstack((std_pos, std_pos, std_vel, std_vel))

        motion_cov = np.einsum('ij,ik->ijk', std, std)

        mean = mean @ self._motion_mat.T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8-dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4-dimensional measurement vector (x, y, w, h), where (x, y)
            is the center position, w the width, and h the height of the
            bounding box.

        Returns
        -------
        mean : ndarray
            The measurement-corrected state mean vector.
        covariance : ndarray
            The measurement-corrected state covariance matrix.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        mean += np.dot(innovation, kalman_gain.T)
        covariance -= np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return mean, covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8-dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('Invalid distance metric')
