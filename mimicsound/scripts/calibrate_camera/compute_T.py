import numpy as np

# translation
t = np.array([-0.481, 0.011, 0.705]) # left

roll  = 2.9
pitch = -0.083
yaw   = 0.011

# t = np.array([0.428, 0.049, 0.743]) # right

# roll  = -2.817
# pitch = -0.042
# yaw   = 0.047 # right

# Rotation matrices for intrinsic XYZ order (roll-pitch-yaw)
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll),  np.cos(roll)]
])
Ry = np.array([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
])
Rz = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw),  np.cos(yaw), 0],
    [0, 0, 1]
])

# Combined rotation
R = Rz @ Ry @ Rx

# Homogeneous transform
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = t
print(T)  

# left:
# [[ 0.99649719 -0.00915338  0.08312385 -0.481     ]
#  [ 0.01096191 -0.9711176  -0.23834941  0.011     ]
#  [ 0.08290473  0.23842571 -0.96761562  0.705     ]
#  [ 0.          0.          0.          1.        ]]

# right:
# [[ 0.99801481,  0.05790533,  0.02476729 , 0.428     ],
#  [ 0.04694127, -0.94610495,  0.32044023 , 0.049     ],
#  [ 0.04198765, -0.31864149, -0.9469449 ,  0.743     ],
#  [ 0.     ,     0.     ,     0.     ,     1.        ]]