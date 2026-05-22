import numpy as np
def generate_noisy_measurements(true_states, noise_std_dev=150.0):
    """
    Simulates the imperfect hardware sensors by ading random Gaussian noise
    to satellite's true position.
    
    Params:
    true_states: Array of true state vectors (shape: [6, N])
    measurement_noise_std: Standard deviation of measurement noise
    
    Returns:
    measurements: Noisy measurements (shape: [6, N])
    """
    true_positions = true_states[0:3, :]
    noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=true_positions.shape)
    noisy_measurements = true_positions + noise
    return noisy_measurements
class KalmanFilter:
    def __init__(self, initial_state, noise_std_dev, dt):
        """
        Initializes the 6-DoF Kalman Filter.
        """
        # State Vector: [x, y, z, vx, vy, vz]
        self.state = initial_state
        
        # P: Covariance Matrix (How confident are we in our current state?)
        # We start with a high number (10.0) because we are initially uncertain.
        self.P = np.eye(6) * 10.0 
        
        # F: State Transition Matrix (Kinematics)
        # Calculates new position based on current velocity and time step (dt)
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt
        
        # H: Observation Matrix
        # Maps our 6-DoF state to our 3-DoF sensors (We only measure X, Y, Z, NOT velocity)
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        
        # R: Measurement Noise Covariance (How much we trust the hardware)
        # Variance is Standard Deviation squared!
        self.R = np.eye(3) * (noise_std_dev ** 2)
        
        # Q: Process Noise Covariance (How much we trust our physics math)
        self.Q = np.eye(6) * 0.1 

    def predict(self):
        """Step 1: Predict the next state using kinematics."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """Step 2: Correct the prediction using the noisy sensor reading."""
        # 1. Calculate the Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 2. Calculate the "Innovation" (Difference between sensor and prediction)
        y = measurement - (self.H @ self.state)
        
        # 3. Update the true state
        self.state = self.state + (K @ y)
        
        # 4. Update the covariance matrix (Uncertainty shrinks!)
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        # Return the filtered X, Y, Z position
        return self.state[0:3]
    