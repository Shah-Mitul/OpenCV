import cv2
import numpy as np
import time
import csv

# Initialize the video capture object
cap = cv2.VideoCapture("D:\\Images\\Random.mp4")

# Initialize an empty white image for collisions
img = np.ones([360, 640, 3], np.uint8) * 255

# Variables for tracking the ball and collisions
previous_position = None
previous_velocity = [0, 0]
velocity = [0, 0]
previous_time = time.time()
collision_time = 0

# Define color bounds for the ball in HSV
lower_color = np.array([20, 100, 100])
upper_color = np.array([30, 255, 255])

# Collision detection parameters
velocity_threshold = 50
collision_detected = False
collision_buffer_time = 2.5  # Buffer time to avoid multiple detections

count = 0

# Frame dimensions and desired scaled ranges
frame_width, frame_height = 640, 360
x_range, y_range = 24, 8

# Open a CSV file to store collision coordinates
with open("Random.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Collision Number", "X Coordinate", "Y Coordinate"])  # CSV header

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        # Resize the frame
        frame = cv2.resize(frame, (640, 360))

        # Convert to HSV and create a mask for the ball's color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 10:
                # Draw the circle and calculate velocity
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                current_time = time.time()
                time_diff = current_time - previous_time

                if time_diff > 0 and previous_position is not None:
                    velocity[0] = (x - previous_position[0]) / time_diff
                    velocity[1] = (y - previous_position[1]) / time_diff
                    velocity_change = np.sqrt((velocity[0] - previous_velocity[0])**2 + (velocity[1] - previous_velocity[1])**2)
                    
                    # Check for collisions
                    if (velocity_change > velocity_threshold and 15 < radius < 30):
                        if not collision_detected and (current_time - collision_time > collision_buffer_time):
                            collision_detected = True
                            collision_time = current_time
                            count += 1

                            # Scale coordinates
                            scaled_x = (x / frame_width) * x_range
                            scaled_y = ((frame_height - y) / frame_height) * y_range  # Invert y to start from the bottom

                            # Draw the collision circle and text
                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                            cv2.putText(frame, f"Collision {count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, f"x = {scaled_x:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, f"y = {scaled_y:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            # Save collision data to CSV
                            writer.writerow([count, scaled_x, scaled_y])

                            # Update the img with detected collisions at scaled coordinates
                            collision_x = int(scaled_x * (640 / x_range))  # Scale x for img
                            collision_y = int((y_range - scaled_y) * (360 / y_range))  # Invert y for img
                            img = cv2.circle(img.copy(), (collision_x, collision_y), int(radius), (0, 0, 255), -1)
                            img = cv2.putText(img.copy(), f"({scaled_x:.2f},{scaled_y:.2f})",
                                              (collision_x, collision_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
                            cv2.waitKey(100)
                    else:
                        collision_detected = False

                    # Debug output
                    print(f"Time since last collision: {current_time - collision_time:.2f}s")
                    print(f"Velocity change: {velocity_change:.2f}")
                    print(f"Collision detected: {collision_detected}")

                # Update tracking variables
                previous_position = [x, y]
                previous_velocity = velocity.copy()
                previous_time = current_time

        # Show the frames
        cv2.imshow("Frame", frame)
        cv2.imshow("Collision Image", img)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Save the final collision image
cv2.imwrite("Collisions.jpg", img)

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

print(f"Total Collisions Detected: {count}")
