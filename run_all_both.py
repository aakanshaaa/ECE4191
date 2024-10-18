# Imports
import time
import RPi.GPIO as GPIO # importing module to work with the raspberry pis GPIO pins
import cv2
import numpy as np
from scipy.interpolate import griddata, Rbf
import os

# Define GPIO pins
motor_A_in2 = 24  # Right motor
motor_A_in1 = 25
motor_A_en = 19

motor_B_in1 = 18  # Left motor
motor_B_in2 = 23
motor_B_en = 13

motor_C = 4


# Set GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_A_in1, GPIO.OUT)
GPIO.setup(motor_A_in2, GPIO.OUT)
GPIO.setup(motor_B_in1, GPIO.OUT)
GPIO.setup(motor_B_in2, GPIO.OUT)
GPIO.setup(motor_A_en, GPIO.OUT)
GPIO.setup(motor_B_en, GPIO.OUT)
GPIO.setup(motor_C, GPIO.OUT)

# Define GPIO pins 
LEFT_ENCODER_A_PIN = 2
LEFT_ENCODER_B_PIN = 3
RIGHT_ENCODER_A_PIN = 17
RIGHT_ENCODER_B_PIN = 27

# Set up the GPIO pins for encoders
GPIO.setup(LEFT_ENCODER_A_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LEFT_ENCODER_B_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(RIGHT_ENCODER_A_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(RIGHT_ENCODER_B_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initiliase parameters
position_left = 0
position_right = 0

last_time_left = time.time()
last_time_right = time.time()

#Calibration Factors
gear_ratio = 75
wheel_circumference = 17.4 #in cm 
ticks_per_rev = 48
distance_per_tick = wheel_circumference / ((ticks_per_rev/2)*gear_ratio)

# Left Encoder callback function
def encoder_callback_left(channel):
    global position_left, last_time_left

    # Read the current state of the left encoder
    LEFT_A_state = GPIO.input(LEFT_ENCODER_A_PIN)
    LEFT_B_state = GPIO.input(LEFT_ENCODER_B_PIN)

    # Determine the direction of the rotation of left wheel
    if LEFT_A_state == LEFT_B_state:
        position_left += distance_per_tick
    else:
        position_left -= distance_per_tick



#Right Encoder callback function
def encoder_callback_right(channel):
    global position_right, last_time_right

    # Read the current state of the right encoder
    RIGHT_A_state = GPIO.input(RIGHT_ENCODER_A_PIN)
    RIGHT_B_state = GPIO.input(RIGHT_ENCODER_B_PIN)

    # Determine the direction of the rotation of right wheel
    if RIGHT_A_state == RIGHT_B_state:
        position_right += distance_per_tick
    else:
        position_right -= distance_per_tick


# Set up the interrupts for both the encoders
GPIO.add_event_detect(LEFT_ENCODER_A_PIN, GPIO.BOTH, callback=encoder_callback_left)
GPIO.add_event_detect(RIGHT_ENCODER_A_PIN, GPIO.BOTH, callback=encoder_callback_right)


# Motor control functions
def control_motor(motor, direction):
    if motor == 'A':  # Right motor
        if direction == 'forward':
            GPIO.output(motor_A_in1, GPIO.LOW)
            GPIO.output(motor_A_in2, GPIO.HIGH)
        elif direction == 'backward':
            GPIO.output(motor_A_in1, GPIO.HIGH)
            GPIO.output(motor_A_in2, GPIO.LOW)
    elif motor == 'B':  # Left motor
        if direction == 'forward':
            GPIO.output(motor_B_in1, GPIO.HIGH)
            GPIO.output(motor_B_in2, GPIO.LOW)
        elif direction == 'backward':
            GPIO.output(motor_B_in1, GPIO.LOW)
            GPIO.output(motor_B_in2, GPIO.HIGH)
    elif motor == 'C':
        GPIO.output(motor_C, GPIO.HIGH)

# Enable motors with PWM and control speed
def enable_motor_pwm(motor, speed):
    if motor == 'A':  # Right motor
        pwm_R = GPIO.PWM(motor_A_en, 10)  # 10 Hz frequency
        pwm_R.start(0)
        pwm_R.ChangeDutyCycle(speed)
        return pwm_R
    elif motor == 'B':  # Left motor
        pwm_L = GPIO.PWM(motor_B_en, 10)  # 10 Hz frequency
        pwm_L.start(0)
        pwm_L.ChangeDutyCycle(speed)
        return pwm_L

# Stop all motors by setting enable pins to LOW
def stop_all_motors(pwm_instance):
    pwm_instance.stop()
    GPIO.output(motor_A_en, GPIO.LOW)
    GPIO.output(motor_B_en, GPIO.LOW)

# Spin motor C to deposit a tennis ball
def drop(ball_dist): 

    # Speed
    speed = 100

    ball_dist = ball_dist*100 #Value is passed in m, converts it into cm
    # Calculate time to turn on motors
    d = ball_dist #[cms]
    v = 26.4 #[cm/s]
    overshoot = 11.3 #[cm]
    t = (d - overshoot) / v # [s]


    # Start motors moving forward

    control_motor('C', 'forward')
    print("[motors started]")

    # Sleep time
    time_motor_on = t
    time.sleep(time_motor_on)
    time.sleep(1.5) #Ensure that the robot / motor has stopped motion

    # Stop motors 

    GPIO.output(motor_C, GPIO.LOW)
    print("[motors stopped]")

# Turns on all motors for ball collection
def move_forward_collect(ball_dist, x, y, robot_angle): 

    # Speed
    speed = 100

    ball_dist = ball_dist*100 #Value is passed in m, converts it into cm
    # Calculate time to turn on motors
    d = ball_dist #[cms]
    v = 26.4 #[cm/s]
    overshoot = 11.3 #[cm]
    t = (d - overshoot) / v # [s]

    # Enable motor pwm
    pwm_L = enable_motor_pwm('A', speed)
    pwm_R = enable_motor_pwm('B', speed)
    print("[motors enabled]")

    # Start motors moving forward
    control_motor('A', 'forward')
    control_motor('B', 'forward')
    control_motor('C', 'forward')
    print("[motors started]")

    # Sleep time
    time_motor_on = t
    time.sleep(time_motor_on)
    time.sleep(1.5) #Ensure that the robot / motor has stopped motion

    # Stop motors 
    stop_all_motors(pwm_L)
    stop_all_motors(pwm_R)
    GPIO.output(motor_C, GPIO.LOW)
    print("[motors stopped]")

    # Additional information
    print(f"Position after {time_motor_on}[s]: {position_left}[cms], {position_right}[cms]")
    
    # Find distance travelled
    mean_distance = (position_left + position_right)/2
    
    # Find new position & angle
    x += np.cos(np.deg2rad(robot_angle))*mean_distance #robot angle?
    y += np.sin(np.deg2rad(robot_angle))*mean_distance

    return x,y

# drives forward
def move_forward(ball_dist, x, y, robot_angle): 

    # Speed
    speed = 100

    ball_dist = ball_dist*100 #Value is passed in m, converts it into cm
    # Calculate time to turn on motors
    d = ball_dist #[cms]
    v = 26.4 #[cm/s]
    overshoot = 11.3 #[cm]
    t = (d - overshoot) / v # [s]

    # Enable motor pwm
    pwm_L = enable_motor_pwm('A', speed)
    pwm_R = enable_motor_pwm('B', speed)
    print("[motors enabled]")

    # Start motors moving forward
    control_motor('A', 'forward')
    control_motor('B', 'forward')
    print("[motors started]")

    # Sleep time
    time_motor_on = t
    time.sleep(time_motor_on)
    time.sleep(1.5) #Ensure that the robot / motor has stopped motion

    # Stop motors 
    stop_all_motors(pwm_L)
    stop_all_motors(pwm_R)
    print("[motors stopped]")

    # Additional information
    print(f"Position after {time_motor_on}[s]: {position_left}[cms], {position_right}[cms]")
    
    # Find distance travelled
    mean_distance = (position_left + position_right)/2
    
    # Find new position & angle
    x += np.cos(np.deg2rad(robot_angle))*mean_distance #robot angle?
    y += np.sin(np.deg2rad(robot_angle))*mean_distance

    return x,y

# drives backward
def move_backward(ball_dist, x, y, robot_angle): 

    # Speed
    speed = 100

    ball_dist = ball_dist*100 #Value is passed in m, converts it into cm
    # Calculate time to turn on motors
    d = ball_dist #[cms]
    v = 26.4 #[cm/s]
    overshoot = 11.3 #[cm]
    t = (d - overshoot) / v # [s]

    # Enable motor pwm
    pwm_L = enable_motor_pwm('A', speed)
    pwm_R = enable_motor_pwm('B', speed)
    print("[motors enabled]")

    # Start motors moving forward
    control_motor('A', 'backward')
    control_motor('B', 'backward')
    print("[motors started]")

    # Sleep time
    time_motor_on = t
    time.sleep(time_motor_on)
    time.sleep(1.5) #Ensure that the robot / motor has stopped motion

    # Stop motors 
    stop_all_motors(pwm_L)
    stop_all_motors(pwm_R)
    print("[motors stopped]")

    # Additional information
    print(f"Position after {time_motor_on}[s]: {position_left}[cms], {position_right}[cms]")
    
    # Find distance travelled
    mean_distance = (position_left + position_right)/2
    
    # Find new position & angle
    x += np.cos(np.deg2rad(robot_angle))*mean_distance #
    y += np.sin(np.deg2rad(robot_angle))*mean_distance

    return x,y

#rotates robot
def spin(ball_angle, x, y, robot_angle):

    # Speed
    speed = 100

    # Calculate time to turn on motors
    d = 100 #[cms]  #distance required to move each motor by a certain angle
    v = 26.4 #[cm/s]
    overshoot = 11.3 #[cm]
    t = (d - overshoot) / v # [s]

    # Enable motor pwm
    pwm_L = enable_motor_pwm('A', speed)
    pwm_R = enable_motor_pwm('B', speed)
    print("[motors enabled]")

    
    if ball_angle > 0:   #Spin left
        control_motor('A', 'backward')
        control_motor('B', 'forward')
    	
    elif ball_angle < 0: #Spin right
        control_motor('A', 'forward')
        control_motor('B', 'backward')
    	
    		
    # Sleep time
    time_motor_on = t
    time.sleep(time_motor_on)
    time.sleep(1.5) #Ensure that the robot / motor has stopped motion

    # Stop motors 
    stop_all_motors(pwm_L)
    stop_all_motors(pwm_R)
    print("[motors stopped]")

    # Additional information
    print(f"Position after {time_motor_on}[s]: {position_left}[cms], {position_right}[cms]")
    
    # Find distance travelled
    mean_distance = (position_left + position_right)/2
    
    # Find new angle
    robot_angle += ball_angle

    return robot_angle
# ----- Code from CV stuff -----
def capture_img(camera_index = 0):
    global cap

    cap = cv2.VideoCapture(camera_index)
   
    # Capture a frame
    ret, frame = cap.read()
    img = frame
    cap.release()
    cv2.destroyAllWindows()

    return img

def dist_from_centre(centrepoint,locations):
    centre_x, centre_y = centrepoint
    distances = []
    for coord in locations:
        distances.append(   np.sqrt( np.square(centre_y - coord[1]) + np.square(centre_x - coord[0]) ))

    return np.array(distances)

def process_img(img): #pass camera coeffs from calibration after
  
  ## Correct distortion - by applying barrel distortion correction

  # Distortion coefficients
  k1 = -0.25  # Radial distortion coefficient
  k2 = 0.0   # Radial distortion coefficient
  p1 = 0.0   # Tangential distortion coefficient
  p2 = 0.0   # Tangential distortion coefficient

  # Camera matrix (assuming no skew, focal lengths are equal, and the principal point is at the center)
  h, w = img.shape[:2]
  camera_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)

  # Distortion coefficients
  dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)

  # Get the new optimal camera matrix
  new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

  # Undistort the grayscale image
  corrected_image = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

  # Crop the image based on the ROI (not sure?)
  x, y, w, h = roi
  corrected_image = corrected_image[y:y+h, x:x+w]

  
  return corrected_image

def locate_tennis_ball(img):
    
    middle_bottom_pixel = []
    tennis_ball_rgb = [223,255,79]

    # Convert the image to hsv
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  
    # Extract the H, S, and V channels
    h_channel, s_channel, v_channel = cv2.split(image_hsv)

    # Normalize the H and S channels to a range of 0-1 for comparison
    h_channel_norm = h_channel / 180.0  # Hue is in the range [0, 180] in OpenCV
    s_channel_norm = s_channel/ 255.0

    # Convert the tennis ball RGB to HSV
    tennis_ball_hsv = cv2.cvtColor(np.uint8([[tennis_ball_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

    # Normalize the tennis ball HSV value
    tennis_ball_hue_norm = tennis_ball_hsv[0] / 180.0
    tennis_ball_saturation_norm = tennis_ball_hsv[1]/ 255.0

    # Define tolerance
    hue_tolerance = 0.1
    saturation_tolerance = 0.8

    # Create a mask for pixels within the tolerance of the tennis ball hue and saturation
    mask = (
        (np.abs(h_channel_norm - tennis_ball_hue_norm) <= hue_tolerance) &
        (np.abs(s_channel_norm - tennis_ball_saturation_norm) <= saturation_tolerance)
    )

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_image = np.zeros_like(grey_img)
    masked_image[mask] = grey_img[mask]
   
    blurred_image=  cv2.blur(grey_img, (3, 3))
  
    detected_circles = cv2.HoughCircles(blurred_image,
                        cv2.HOUGH_GRADIENT_ALT, 2, 30, param1 = 300,
                    param2 = 0.5, minRadius = 1, maxRadius = -1)

    # Draw circles that are detected.
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.circle(img, (a, b+r), 1, (255, 0, 0), 3)
            middle_bottom_pixel.append([a, b+r])
            
    return np.array(middle_bottom_pixel)

def dist_rot_to_target(target_location): 
    
    target_locations = np.array(target_locations)

    # homography for pixel to world coordinate conversion
    h = np.array([[0.253740610646319,1.32996851572853,-2122.36980102190],
    [1.19833484185468,0.00788378185419525,-762.669172803865],
    [0.103406288700081,-6.15470279781376,597.349215854788]])
    #
    image_coords = np.hstack([target_locations,np.ones([target_locations.shape[0],1])])

    world_coords = h@image_coords.T
    x = np.divide(world_coords[0,:],world_coords[2,:])
    y = np.divide(world_coords[1,:],world_coords[2,:])

    dist = np.round(np.sqrt(np.square(x)+np.square(y)),2)
    angle= np.round(np.degrees(np.arctan2(y,x)),2)
    
    return dist, angle

def locate_tennis_box_mask(img):
    
    middle_bottom_pixel = []
    box_rgb = [165, 150,100]#nnis_ball_rgb = [223,255,79] [160, 158, 56]
    #mask for green first

    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract the H, S, and V channels
    h_channel, s_channel, v_channel = cv2.split(image_hsv)

    # Normalize the H and S channels to a range of 0-1 for comparison
    h_channel_norm = h_channel / 180.0  # Hue is in the range [0, 180] in OpenCV
    s_channel_norm = s_channel/ 255.0

    # Convert the tennis ball RGB to HSV
    box_hsv = cv2.cvtColor(np.uint8([[box_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    # tennis_ball_hsv[1] = np.clip(tennis_ball_hsv[1]*5,0,255)

    # Normalize the tennis ball HSV value
    box_hue_norm = box_hsv[0] / 180.0
    box_saturation_norm = box_hsv[1]/ 255.0

    # Define tolerance
    hue_tolerance = 1/30
    saturation_tolerance = 0.6

    # Create a mask for pixels within the tolerance of the tennis ball hue and saturation
    mask = (
        (np.abs(h_channel_norm - box_hue_norm) <= hue_tolerance) &
        (np.abs(s_channel_norm - box_saturation_norm) <= saturation_tolerance)
    )

    
    grey_img = cv2.blur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(15,15))
    masked_image = np.zeros_like(grey_img)
    masked_image[mask] = grey_img[mask]
    
    blurred_image=  cv2.blur(masked_image, (15,15))

    return blurred_image#np.array(middle_bottom_pixel)

def find_box(img_mask):
    ret,thresh = cv2.threshold(img_mask,50,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print("Number of contours detected:",len(contours))
    XN=0
    YN=0
    CN=0
    arc=0.01
    for cnt in contours:
            XN=0
            YN=0
            CN=0
            #approx= cv2.approxPolyN(7,
            approx = cv2.approxPolyDP(cnt, arc*cv2.arcLength(cnt, True), True)
            #approx = cv2.approxPolyDP(7, 0.01*cv2.arcLength(cnt, True), True)
            for i in range(len(approx)):
                XN=XN+approx[i][0][0]
                YN=YN+approx[i][0][1]
                CN=CN+1
            (x,y)=cnt[0,0]
            print(CN)
            if len(approx) >= 10:
                arc=arc+0.005
            if len(approx) >= 5:
                img = cv2.drawContours(img, [approx], -1, (0,0,255), 8)

    x_values = [x[0] for x in approx]
    x_average = sum(x_values) / len(x_values)
    

    centreline = len(img_mask[0])/2
    print(centreline)
    box_found = False

    margin = 100  
    if(sum(img_mask)>100):
        print("too close") 

        #move a bit closer and deposit ball
        x,y = move_forward(0.05,0,0,0)
        drop()
        box_found = True

    elif int(XN/CN)<(centreline+margin) and int(XN/CN)>(centreline-margin):
        print("print centre")
        box_state = "centre"
        # move forward 0.20m
        x,y = move_forward(0.2,0,0,0)
    
    elif int(XN/CN) > centreline:
        print("print right")
        #rotate right
        ang = spin(-20, 0, 0, 0)
        
    elif int(XN/CN) < centreline:
        print("print left")
        ang = spin(20, 0, 0, 0)
    return box_found

def create_binary_mask(image_shape, upper_point, lower_point):
    # Create a blank mask of zeros (same height and width as the image)
    mask = np.ones(image_shape[:2], dtype=np.uint8)

    # Get the height and width of the image
    height, width = image_shape[:2]

    # Create an array of all the x coordinates (columns)
    x_coords = np.arange(width)

    # Calculate the slope and intercept of the line using the two points
    x1, y1 = upper_point
    x2, y2 = lower_point
    
    if x1 != x2:  # To avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        # If x1 == x2, the line is vertical, set slope to infinity
        slope = np.inf

    # Create the mask where all points below the line are 1, and above are 0
    for x in x_coords:
        if slope != np.inf:
            # y = mx + c (equation of the line)
            y_line = np.min([int(slope * x + intercept),image_shape[0]])
            if y_line <= 0:
                y_line = 0
        else:
            # For a vertical line, y values should simply compare with x1
            y_line = y1

        # All pixels below this line (y > y_line) are set to 1
        mask[0:y_line, x] = 0

    return mask

def linefindhough(img):

    # line_mask = np.zeros(img.shape)
    ## threshold image to line colour
    threshold = 190

    # where img above or between threshold, make white else black
    t_img = np.where(img > threshold,35,0)


    # # Edge detection
    #edges = cv2.Canny(img.astype(np.uint8)*255,190,220)
    edges = cv2.Canny(t_img.astype(np.uint8)*255,20,50)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.deg2rad(1), 150)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    line_mask = np.ones([img_rows,img_cols])
    if lines is not None:
        
        mask_sum = img_rows*img_cols
        max_x = 0
        min_x = img_cols
        max_y = 0
        min_y = img_rows
        line_count = 0
        for r_theta in lines:
            if line_count <=3:
                arr = np.array(r_theta[0], dtype=np.float64)
                r, theta = arr
                # Stores the value of cos(theta) in a
                a = np.cos(theta)

                # Stores the value of sin(theta) in b
                b = np.sin(theta)

                # x0 stores the value rcos(theta)
                x0 = a*r

                # y0 stores the value rsin(theta)
                y0 = b*r

                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1000*(-b))

                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1000*(a))

                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1000*(-b))

                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1000*(a))

                if y1 < y2:
                    upper_point = [x2,y2]
                    lower_point = [x1,y1]
                else:
                    lower_point = [x2,y2]
                    upper_point = [x1,y1]

                temp_line_mask = create_binary_mask(img.shape,upper_point,lower_point)
                if np.sum(temp_line_mask) < mask_sum:
                    mask_sum = np.sum(temp_line_mask)
                    line_mask = temp_line_mask
            else:
                break

    return line_mask

def update_robot_position(t_matrix,dist = 0, rot =0):
    
    ## assume robot starts at origin of base with axes aligned
    ## rotations are only about z and translation is only in x direction
    rot = np.deg2rad(rot)

    t_last = np.array([[np.cos(rot),-np.sin(rot),dist],
            [np.sin(rot),np.cos(rot),0],
            [0,0,1]])
    t_matrix = t_matrix@t_last
    
    return t_matrix
   
# ----- RUN CODE -----

# Main testing loop
def goal_machine():

    # intialising variables
    rot_threshold = 3 #in degs
    ball_collected = False
    ball_locations = np.array([])
    goal='ball'
    robot_x = 0
    robot_y = 0
    robot_angle = 0
    inf_loop_check1 = 0
    inf_loop_check2 = 0
     
    t_matrix = np.eye(3) #may want to change to 90 rotation as initial position
    t_matrix =  update_robot_position(t_matrix,dist = 0, rot = np.deg2rad(90)) #starts 90 degrees from base x axis
    
    while True: #continuously run loop until break
        if goal == 'ball': # while locating and navigating to the ball

            inf_loop_check1+=1
            if inf_loop_check1 > 10:
                print("Can't navigate to ball")
                return

            rotation_search = 0 # for rotating robot to find balls - at start find balls on spot
            distance_search = 0 # for rotating robot to find balls - at start find balls on spot
            
            total_dist = 0 #track the distance travelled searching for ball before rotating
            total_rot = 0 # track rotation while finding a ball
            max_distance_search = 2 #metres travelled before rotating
            
            while ball_locations.shape[0]<1: # find tennis ball if ball locations is none
    
                #rotate and move robot by rotation and distance variable (initially 0)
                #move_robot(rot,dist)
                robot_angle = spin(rotation_search, robot_x, robot_y, robot_angle)
                robot_x,robot_y = move_forward(distance_search , robot_x, robot_y, robot_angle)
                t_matrix = update_robot_position(t_matrix,dist = distance_search, rot =rotation_search) 

                total_rot += rotation_search # track total rotation
                total_dist += distance_search
                distance_search = 1 #if the loop runs again, the robot will move the specified distance
                rotation_search = 0 #just reset the search distance every iteration until it reaches dist_threshold
                
                # Capture an image
                img = capture_img(0)

                # Preprocess image (correct for distortion [in bgr])
                img_processed = process_img(img)

                # Get mask for valid court boundaries
                line_mask = linefindhough(img_processed)

                # Find ball location in pixels (segment, find boxes, some additional logic)
                ball_locations = locate_tennis_ball(img_processed)

                if total_dist >= max_distance_search:
                    distance_search = 0 # stop moving until ball found or rotations complete
                    rotation_search =  90 #rotate until ball found or full spin

                    if total_rot >= 360:
                        total_dist = 0 #reset total distance
                        total_rot = 0 #reset total rotation
                        distance_search = 1 #keep going forward
                        rotation_search = 0 #no more spinning until max distance reached again
            
                inf_loop_check2+= 1
                if inf_loop_check2 > 20:
                    print("Can't find ball")
                    return
                
                if ball_locations.shape[0]>0:
                    valid_locs = []
                for loc in ball_locations:
                    r,c, = loc[0],loc[1]
                    if line_mask[r,c]:
                        valid_locs.append(loc)

                ball_locations = np.array(valid_locs)
                ## END OF inner loop to find a tennis ball
        
            ## if ball locations - if there is a ball detected
            centrepoint = [img.shape[0]/2,img.shape[1]]
            distances = dist_from_centre(centrepoint,ball_locations)
            ball_locations = ball_locations[np.argsort(distances)] #hardcode to choose the nearest circle
            
            # get distance and rotation (transform pixels into a distance and degrees/rotation)
            dist, rot = dist_rot_to_target(ball_locations[0])

            print('distance:',dist,'rotation:',rot)

            # determine the distance the robot will travel before finding tennis balls again
            if dist >= 1:
                dist = 1 #cap distance travelled to 1m

            elif 0.75 < dist < 1: #can reduce the 0.75 to slightly above detection distance
                dist = 0.5 #only travel 0.5m

            else: #in collecting range
                dist += 0.1 #go extra 10cm to 'collect'
                ball_collected = True

            if abs(rot) > rot_threshold:
                rot = 0

            # pass new dist and rot to robot moving function
            robot_angle = spin(rot, robot_x, robot_y, robot_angle)
            robot_x,robot_y = move_forward(dist , robot_x, robot_y, robot_angle)

            #might change to use actual dist and rot from encoders
            t_matrix = update_robot_position(t_matrix,dist = dist, rot =rot) 
    
            ## return back after ball is 'collected'
            if ball_collected:
                goal = 'box'

            ## END OF OUTER LOOP

        if goal == 'box':
            box_found = False
            while box_found == False:
                box_img = capture_img(0)
                processed_box_img = process_img(box_img)
                img_mask = locate_tennis_box_mask(img_processed)
                box_found = find_box(img_mask)

            # once box found, deposit ball and reverse and rotate so that more balls can be found
            drop()
            robot_x,robot_y = move_backward(0.1,robot_x,robot_y,robot_angle)
            robot_angle = spin(180,robot_x,robot_y,robot_angle)
            goal = 'ball' 



        if goal == 'home': #not fully integrated
        
            ## or find using the continuously updating robot x,y and angle
            distance_home = np.linalg.norm([t_matrix[0,2],t_matrix[1,2]])
            rotation_home = np.degrees(np.arctan2(-t_matrix[1,2],-t_matrix[0,2])- np.arctan2(t_matrix[1,0],t_matrix[0,0])) #double check this
            
            # navigate to start position
            spin(rotation_home, robot_x, robot_y, robot_angle)
            move_forward(distance_home , robot_x, robot_y, robot_angle)

            break
    return


## ive just used my template for now but can swap around
# def run():
    
#     Status = "ball"
    
#     # Initial position and angle set
#     robot_x = 0
#     robot_y = 0
#     robot_angle = 0
    
#     # Enter a loop to progressively find and move towards a tennis ball
#     while Status == "Find_ball":
    	
#     	# Take image 
    	
#     	# find distance and angle to ball
#     	ball_dist, ball_angle = # find_ball() 
    	
#     	if # position of ball detected (find_ball is True: ?)
    	
#     		if # mod(ball_angle) > threshold degrees
    		
#     			#spin robot by angle of ball
#     			spin(ball_angle, robot_x, robot_y, robot_angle)
    			
#     			# move towards ball (distance + 20cm)
#     			move_forward(ball_dist + 20 , robot_x, robot_y, robot_angle)
    		
#     		else # angle < threshold
#     			# move towards ball (distance + 20cm)
#     			move_forward(ball_dist + 20 , robot_x, robot_y, robot_angle)
    		
#     		if # ball found
#     			Status = "Go_home" 
#     			print("ball found!")
    		
#     	else robot_x > 3 or robot_y > 3: # explore (move forward 1m, maybe do a spin if x or y >3m)
#     	move_forward(ball_dist= 1, robot_x, robot_y, robot_angle)    
#     	spin(ball_angle, robot_x, robot_y, robot_angle)
    	
  
#     if Status == "Go_home":
#     	find_home(x,y,angle)  #call find_home function
#     	print("Robot home safe. Mission Complete!")

    	
    
# # Make sure that we can calculate the distance from the start point to the current location
# #def calculate_distance_from_start(robot_x, robot_y):
#     #return math.sqrt(robot_x**2 + robot_y**2)
    	
  
# # Clean up GPIO
# GPIO.cleanup()

            
goal_machine()
GPIO.cleanup()
