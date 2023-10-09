
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
import numpy
import np
import math

# This function returns the angle between a and b
def angle_from_dot_product(a,b):

    ax=a[0]
    ay=a[1]
    az=a[2]

    bx = b[0]
    by = b[1]
    bz = b[2]

    a_mag=np.sqrt(np.power(ax, 2)+np.power(ay, 2)+np.power(az, 2))
    b_mag=np.sqrt(np.power(bx, 2)+np.power(by, 2)+np.power(bz, 2))

    theta=np.arccos( (1/(a_mag*b_mag))*( ax*bx+ay*by+az*bz ) )

    return theta

# This function converts theta,d,a, and alpha into dh parameters
# this is for relateing reference frames to each other
def dh(theta,d,a,alpha):

# First row of dh table
    A11=np.cos(theta)
    A12=-np.cos(alpha)*np.sin(theta)
    A13=np.sin(alpha)*np.sin(theta)
    A14=a*np.cos(theta)

# second row of dh table
    A21=np.sin(theta)
    A22=np.cos(alpha)*np.cos(theta)
    A23=-np.sin(alpha)*np.sin(theta)
    A24=a*np.sin(theta)

# third row of dh table
    A31=0
    A32=np.sin(alpha)
    A33=np.cos(alpha)
    A34=d

# fourth row of dh table
    A41=0
    A42=0
    A43=0
    A44=1
    A=0
    A=np.array([ [A11, A12, A13, A14],  [A21, A22, A23, A24],  [A31, A32, A33, A34], [A41, A42, A43, A44]])

    return A;

# Projection step used in FABRIK
def project_along_vector(x1, y1, z1, x2, y2, z2, L):
	'''
	 Solve for the point px, py, pz, that is
	 a vector with magnitude L away in the direction between point 2 and point 1,
	 starting at point 1
	 Parameters
	 ----------
	 x1, y1, z1 : scalars
		 point 1
	 x2, y2, z2 : scalars
		 point 2
	 L : scalar
		 Magnitude of vector?
	Returns
	-------
	out : array_like
		projected point on the vector
	'''

	# vector from point 1 to point 2
	vx = x2-x1
	vy = y2-y1
	vz = z2-z1
	v = np.sqrt(np.power(vx, 2)+np.power(vy, 2)+np.power(vz, 2))

	ux = vx/v
	uy = vy/v
	uz = vz/v

	# Need to always project along radius
	# Project backwards
	px = x1+L*ux
	py = y1+L*uy
	pz = z1+L*uz

	return np.array([px, py, pz])

# FABRIK algorithim, see the youtube video
# https://www.youtube.com/watch?v=UNoX65PRehA&t=814s for a really good video

def fabrik(l1,l2,l3,x_prev,y_prev,z_prev,x_command,y_command,z_command,tol_limit,max_iterations):

# Base rotation is simply based on angle made within the x-y plane

    q1_prev=np.arctan2(y_prev[3],x_prev[3])
    q1=np.arctan2(y_command,x_command)

    base_rotation=q1-q1_prev # this is the rotation the base must make to get from initial position to the commanded position

    # Base rotation matrix about z
    R_z = np.array([[np.cos(base_rotation), -np.sin(base_rotation), 0.0],
                    [np.sin(base_rotation), np.cos(base_rotation), 0.0],
                    [0.0, 0.0, 1.0]])

    # Rotate the location of each joint by the base rotation
    # This will force the FABRIK algorithim to only solve
    # in two dimensions, else each joint will move as if it has
    # a 3 DOF range of motion
    # print 'inside the fabrik method and x_joints is'
    # print x_joints
    p4 = np.dot(R_z, [x_prev[3], y_prev[3], z_prev[3]])
    p3 = np.dot(R_z, [x_prev[2], y_prev[2], z_prev[2]])
    p2 = np.dot(R_z, [x_prev[1], y_prev[1], z_prev[1]])
    p1 = np.dot(R_z, [x_prev[0], y_prev[0], z_prev[0]])

    # Store the (x,y,z) position of each joint

    p4x = p4[0]
    p4y = p4[1]
    p4z = p4[2]

    p3x = p3[0]
    p3y = p3[1]
    p3z = p3[2]

    p2x = p2[0]
    p2y = p2[1]
    p2z = p2[2]

    p1x = p1[0]
    p1y = p1[1]
    p1z = p1[2]

    # Starting point of each joint
    p1x_o=p1x
    p1y_o=p1y
    p1z_o=p1z

    iterations=0
    for j in range(1,max_iterations+1):

        if np.sqrt(np.power(x_command, 2) + np.power(y_command, 2) + np.power(z_command, 2)) > (l1 + l2 + l3):
            print(' desired point is likely out of reach')


        [p3x, p3y, p3z] = project_along_vector(x_command, y_command, z_command, p3x, p3y, p3z, l3)
        [p2x, p2y, p2z] = project_along_vector(p3x, p3y, p3z, p2x, p2y, p2z, l2)
        [p1x, p1y, p1z] = project_along_vector(p2x, p2y, p2z, p1x, p1y, p1z, l1)

        [p2x,p2y,p2z]=project_along_vector(p1x_o,p1y_o,p1z_o,p2x,p2y,p2z,l1)
        [p3x,p3y,p3z]=project_along_vector(p2x,p2y,p2z,p3x,p3y,p3z,l2)
        [p4x,p4y,p4z]=project_along_vector(p3x,p3y,p3z,x_command,y_command,z_command,l3)

        # check how close FABRIK position is to command position
        tolx = p4x - x_command
        toly = p4y - y_command
        tolz = p4z - z_command

        tol = np.sqrt(np.power(tolx, 2) + np.power(toly, 2) + np.power(tolz, 2))
        iterations = iterations + 1

        # Check if tolerance is within the specefied limit

            # Re-organize points into a big matrix for plotting elsewhere
        p_joints = np.array([[p1x, p2x, p3x, p4x],
                                 [p1y, p2y, p3y, p4y],
                                 [p1z, p2z, p3z, p4z]])


        v21 = np.array([p2x - p1x, p2y - p1y, p2z - p1z])
        v32 = np.array([p3x - p2x, p3y - p2y, p3z - p2z])
        v43 = np.array([p4x - p3x, p4y - p3y, p4z - p3z])


        q2 = np.arctan2((p2z - p1z), np.sqrt(np.power(p2x - p1x, 2) + np.power(p2y - p1y, 2)))


        q3 = -1 * angle_from_dot_product(v21, v32)
        q4 = -1 * angle_from_dot_product(v32, v43)

        q_joints=np.array([q1, q2, q3, q4])

        return q_joints

def forward_kinematics(l1,l2,l3,q1,q2,q3,q4):
    # create inital transformation matricies relateing each joint position to the global fram
    # global frame is the center of the robot

    # T is a 4x4 transformation matrix
    # T(1:3,1:3) is a rotation matrix
    # T(1:3,4) is the x,y,z position of the frame
    # i.e if i want the x position of the second joint it is T(1,4), y position is T(2,4)
    # T(4,:) is used for scaleing and remains at 0 0 0 1 for no scaleing



    T10 = dh(q1, 0, 0, pi / 2)  # Base rotation relative to global frame
    T21 = dh(q2, 0, l1, 0)  # frame of reference at the end of the first link, relative to Base
    T32 = dh(q3, 0, l2, 0)  # second link end relative to first link
    T43 = dh(q4, 0, l3, 0)  # end effector relative to second link, ca nbe considered a third link

    T20 = np.dot(T10, T21)
    T30 = np.dot(T20, T32)
    T40 = np.dot(T30, T43)

    # Create matrix where each row is a joint position, i.e
    # joint positions = [x1 y1 z1
    #                   x2  y2 z2]

    # assume 0 index notation
    # T10=numpy.transpose(T10)
    # print(T10)
    # print(T20)
    p_joints = [T10[0:3, 3], T20[0:3, 3], T30[0:3, 3], T40[0:3, 3]]

    p_joints = np.transpose(p_joints)

    return p_joints

# Define SI Units
# START MAIN PROGRAM
plt.close('all')
fig = plt.figure()

meters=1
feet=0.3048*meters
pi=3.1415
radians=1
degrees=1/360*2*pi*radians

# initial angles of each joint
q1_label='second_master_value'
q1=45*degrees   # Base
q2=60*degrees   # first link relative to base
q3=-30*degrees  # second link relative to first link
q4=-30*degrees  # third link relative to first link

# Define each robot arm link length
l1=3*feet
l2=2*feet
l3=1.5*feet

# Solve for the initial x,y,z points of each joint given joint angles

p_joints=forward_kinematics(l1,l2,l3,q1,q2,q3,q4)


# row matrix of joint x,y,z positions
x_joints_o = p_joints[0]
y_joints_o = p_joints[1]
z_joints_o = p_joints[2]

# row matrix of joint x,y,z positions
x_joints = p_joints[0]
y_joints = p_joints[1]
z_joints = p_joints[2]

x_prev=x_joints_o
y_prev=y_joints_o
z_prev=z_joints_o

x_command=-l2*meters
y_command=-l2*meters
z_command=0

tol_limit=l2/100
max_iterations=10

new_q=fabrik(l1,l2,l3,x_joints_o,y_joints_o,y_joints_o,x_command,y_command,z_command,tol_limit,max_iterations)

# extract each joint angle seperate
q1_new=new_q[0]
q2_new=new_q[1]
q3_new=new_q[2]
q4_new=new_q[3]

# find new x,y,z location of each joint
new_joint_points=forward_kinematics(l1,l2,l3,q1_new,q2_new,q3_new,q4_new)

# extract each joint position into a row matrix
x_IK=new_joint_points[0]
y_IK=new_joint_points[1]
z_IK=new_joint_points[2]

other_value='Alexes value'



style.use('ggplot')
ax1 = fig.add_subplot(111, projection='3d')

# plot original position
ax1.plot(x_joints_o,y_joints_o,z_joints_o,color='blue')
start_joints=ax1.scatter(x_joints_o,y_joints_o,z_joints_o,label='start',color='blue')

# Plot commanded position
IK_position=ax1.scatter(x_IK,y_IK,z_IK,color='red')
ax1.plot(x_IK,y_IK,z_IK,label='IK position',color='red')
command_point=ax1.scatter(x_command,y_command,z_command,color='green')

plt.legend([start_joints,IK_position,command_point], ['Initial Robot Position','Commanded Position','(x_command,y_command,z_command)'])

plt.xlabel("x [m]")
plt.xlabel("y [m]")
plt.xlabel("z [m]")


plt.show()

