#!/usr/bin/env python3
from importlib.resources import open_text
import rospy
import numpy as np
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension
import math

rospy.init_node("state_estimation")

gps_observation = Float64MultiArray()
steering = Float64()
velocity = Float64()

# gps_observation = [0,0,0] # x, y, theta
steering = 0
velocity = 0
dk =0
L =4.9
delta = 0
gps_state = np.array([0,0,0])

# Subscribe to the gps, steering, and velocity topics named below and update the global variables using callbacks
# /gps
# /car_actions/steer
# /car_actions/vel

def update_gps(data):
    global gps_state 
    gps_state = np.array(data.data)

def update_velocity(data):
    global velocity
    velocity = data.data

def update_steering(data):
    global steering
    steering = data.data

rospy.Subscriber("/gps", Float64MultiArray, update_gps )
rospy.Subscriber("/car_actions/steer", Float64, update_steering )
rospy.Subscriber("/car_actions/vel", Float64, update_velocity )

# Publisher for the state
state_pub = rospy.Publisher('vehicle_model/state', Float64MultiArray, queue_size=10)

r = rospy.Rate(10)

# Initialize the start values and matrices here

H_k = np.array([[1.0,  0,   0],  # Measurement matrix
                [  0,1.0,   0],
                [  0,  0, 1.0]])
  


process_noise_v_k_minus_1 = np.array([0.01,0.01,0.003]) # process noise

Q_k = np.array([[1.0,   0,   0],  # the state model noise covariance matrix
                [  0, 1.0,   0],
                [  0,   0, 1.0]])
 
R_k = np.array([[1.0,   0,    0], #Sensor measurement noise covariance matrix
                [  0, 1.0,    0],
                [  0,    0, 1.0]])  

def getA(theta,velocity, dk):
                 
    A= np.array([[1.0,  0,  -velocity *np.sin(theta)*dk],
                 [  0,1.0,  velocity *np.cos(theta)*dk],
                 [  0,  0,  1.0]])
    
    return A

def getB(theta, delta, dk, L, v):
    
    B = np.array([[math.cos(theta)*dk, 0],
                  [math.sin(theta)*dk, 0],
                  [(math.tan(delta)/L)*dk, (v/L)*(1/(math.cos(delta))**2) *dk ]])
    
    return B


def ekf(z_k_observation_vector,   # The observation from the Odometry
        state_estimate_k_minus_1, # The state estimate at time k-1
        control_vector_k_minus_1, # The control vector applied at time k-1
        P_k_minus_1,              # The state covariance matrix estimate at time k-1
        dk):

    z_k_observation_vector = gps_state

    # We want to return the state estimation at time k and the covariance state estimation of time k                        
    # first:
    # Predict the state estimate at time k based on the state... 
    # estimate at time k-1 and the control input applied at time k-1.
    
    state_estimate_k = ((getA(state_estimate_k_minus_1[2],velocity,dk)) @ (state_estimate_k_minus_1))
    + ((getB(state_estimate_k_minus_1[2],delta,dk, L, velocity)) @ (control_vector_k_minus_1)) 
    + (process_noise_v_k_minus_1)               
    # print(f'State Estimate Before EKF={state_estimate_k}')
            
    # Predict the state covariance estimate based on the previous covariance and some noise
    P_k = (getA(state_estimate_k_minus_1[2],velocity,dk)) @ P_k_minus_1 @ (getA(state_estimate_k_minus_1[2],velocity,dk)) + (Q_k) 
                
    # second:
    # Calculate the difference between the actual and predicted measurements                 
    measurement_residual_y_k = (z_k_observation_vector) - ((H_k @ state_estimate_k))

    # Calculate the measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k               
            
    # Calculate the near-optimal Kalman gain
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
            
    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
            
    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)
            
    # Print the best (near-optimal) estimate of the current state of the robot
    # print(f'State Estimate After EKF={state_estimate_k}')

    # Return the updated state and covariance estimates
    return state_estimate_k, P_k
    
dk = 0.5     # 1 second

car_state = np.array([0,0,0])
state_estimate_k_minus_1 = car_state
control_vector_k_minus_1 = np.array([velocity,steering])

P_k_minus_1 = np.array([[0.1,  0,   0],
                        [  0,0.1,   0],
                        [  0,  0, 0.1]])

while not rospy.is_shutdown():

# Create the Kalman Filter here to estimate the vehicle's x, y, and theta
      
    # Run the Extended Kalman Filter and store the 
    # near-optimal state and covariance estimates
    optimal_state_estimate_k, covariance_estimate_k = ekf(
        gps_observation, # Most recent sensor measurement
        state_estimate_k_minus_1, # Our most recent estimate of the state
        control_vector_k_minus_1, # Our most recent control input
        P_k_minus_1, # Our most recent state covariance matrix
        dk) # Time interval
        
    # Get ready for the next timestep by updating the variable values
    state_estimate_k_minus_1 = optimal_state_estimate_k
    P_k_minus_1 = covariance_estimate_k
        
    # Create msg to publish#
    current_state = Float64MultiArray()
    layout = MultiArrayLayout()
    dimension = MultiArrayDimension()
    dimension.label = "current_state"
    dimension.size = 3
    dimension.stride = 3
    layout.data_offset = 0
    layout.dim = [dimension]
    current_state.layout = layout
    current_state.data = optimal_state_estimate_k

    state_pub.publish(current_state)
    r.sleep()