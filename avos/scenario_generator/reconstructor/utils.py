import numpy as np


def update_vel_acc_jerk(timestamp, center_x, center_y, heading):
    N = len(center_x)

    # Compute velocity using np.gradient (first-order derivative)
    velocity_x = np.gradient(center_x, timestamp, edge_order=2)
    velocity_y = np.gradient(center_y, timestamp, edge_order=2)

    # Compute acceleration using np.gradient (second-order derivative)
    acceleration_x = np.gradient(velocity_x, timestamp, edge_order=2)
    acceleration_y = np.gradient(velocity_y, timestamp, edge_order=2)

    # Compute jerk using np.gradient (third-order derivative)
    jerk_x = np.gradient(acceleration_x, timestamp, edge_order=2)
    jerk_y = np.gradient(acceleration_y, timestamp, edge_order=2)

    # Now convert the velocity, acceleration, and jerk to the vehicle coordinate system
    velocity_local_x = np.zeros(N)
    velocity_local_y = np.zeros(N)
    acceleration_local_x = np.zeros(N)
    acceleration_local_y = np.zeros(N)
    jerk_local_x = np.zeros(N)
    jerk_local_y = np.zeros(N)

    for i in range(N):
        # Extract cos and sin of heading for current timestamp
        cos_theta = np.cos(heading[i])
        sin_theta = np.sin(heading[i])

        # Convert velocity from global to local coordinates
        velocity_local_x[i] = velocity_x[i] * cos_theta + velocity_y[i] * sin_theta
        velocity_local_y[i] = -velocity_x[i] * sin_theta + velocity_y[i] * cos_theta

        # Convert acceleration from global to local coordinates
        acceleration_local_x[i] = acceleration_x[i] * cos_theta + acceleration_y[i] * sin_theta
        acceleration_local_y[i] = -acceleration_x[i] * sin_theta + acceleration_y[i] * cos_theta

        # Convert jerk from global to local coordinates
        jerk_local_x[i] = jerk_x[i] * cos_theta + jerk_y[i] * sin_theta
        jerk_local_y[i] = -jerk_x[i] * sin_theta + jerk_y[i] * cos_theta

    return velocity_local_x, velocity_local_y, acceleration_local_x, acceleration_local_y, jerk_local_x, jerk_local_y