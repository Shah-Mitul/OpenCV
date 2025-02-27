import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import time
import csv



# Variables for tracking ball and collisions
previous_position = None
previous_velocity = [0, 0]
velocity = [0, 0]
previous_time = time.time()
collision_time = 0

# Define color bounds for the ball in HSV
lower_color = np.array([20, 100, 100])
upper_color = np.array([30, 255, 255])

# Collision detection parameters
velocity_threshold = 100
collision_detected = False
collision_buffer_time = 1.0

frame_width, frame_height = 640, 360
x_range, y_range = 24, 8



#---------------------------Round 1---------------------------------------------- 

cap = cv2.VideoCapture("D:\\Images\\Imp80.mp4") #Enter video of ball hitting wall 
img = cv2.imread("D:\\Images\\goalpost.jpg")
img = cv2.resize(img, (640, 360))
count = 0

with open("D:\\output.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Collision Number", "X Coordinate", "Y Coordinate"])

    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.putText(frame, f"Shots {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frame = cv2.resize(frame, (640, 360))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 1:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                current_time = time.time()
                time_diff = current_time - previous_time

                if time_diff > 0 and previous_position is not None:
                    velocity[0] = (x - previous_position[0]) / time_diff
                    velocity[1] = (y - previous_position[1]) / time_diff
                    velocity_change = np.sqrt((velocity[0] - previous_velocity[0])**2 + (velocity[1] - previous_velocity[1])**2)

                    if (velocity_change > velocity_threshold and 2 < radius < 15):
                        if not collision_detected and (current_time - collision_time > collision_buffer_time):
                            collision_detected = True
                            collision_time = current_time
                            count += 1

                            scaled_x = (x / frame_width) * x_range
                            scaled_y = ((frame_height - y) / frame_height) * y_range  

                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                            
                            writer.writerow([count, scaled_x, scaled_y])

                            collision_x = int(scaled_x * (640 / x_range))
                            collision_y = int((y_range - scaled_y) * (360 / y_range))
                            img = cv2.circle(img.copy(), (collision_x, collision_y), int(radius), (0, 0, 255), -1)

                    else:
                        collision_detected = False

                previous_position = [x, y]
                previous_velocity = velocity.copy()
                previous_time = current_time

        cv2.imshow("Frame", frame)
        cv2.imshow("Collisions", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cv2.imwrite("D:\\Collisions.jpg", img)
cap.release()
cv2.destroyAllWindows()

print(f"Total Collisions Detected: {count}")



# ----------------------------- Heatmap Calculation--------------------------------


x_div = np.arange(0.00, 24.01, 0.05) 
y_div = np.arange(0.00, 8.01, 0.05)
X, Y = np.meshgrid(x_div, y_div)
Z = np.zeros(X.shape, dtype=np.float32)

data = pd.read_csv("D:\\output.csv")
coordinates = list(zip(data['X Coordinate'].round(2), data['Y Coordinate'].round(2)))

def find_within_radius(center, radius):
    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
q    return np.argwhere(distances <= radius)

for element in coordinates:
    for points, increment in [(find_within_radius(element, 0.5), 7),
                              (find_within_radius(element, 0.85), 5),
                              (find_within_radius(element, 2.00), 1)]:
        for i, j in points:
            Z[i, j] += increment

data["Intensity"] = [Z[np.argmin(np.abs(y_div - y)), np.argmin(np.abs(x_div - x))] for x, y in coordinates]
data.to_csv("D:\\output_updated.csv", index=False)

zero_intensity_indices = np.argwhere(Z == 0)
zero_points = np.array([[x_div[j], y_div[i]] for i, j in zero_intensity_indices])

if zero_points.size > 0:
    num_clusters = 10  
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(zero_points)
    cluster_centers = kmeans.cluster_centers_
    
    weakspots_df = pd.DataFrame(cluster_centers, columns=["X Coordinate", "Y Coordinate"])
    weakspots_df.to_csv("D:\\weakspots.csv", index=False)

    plt.figure(figsize=(24, 8), dpi=100)
    plt.imshow(Z, cmap='jet', interpolation='gaussian', origin='lower', aspect='equal', extent=[0, 24, 0, 8])
    plt.colorbar()

    for center in cluster_centers:
        plt.scatter(center[0], center[1], color='green', s=50, marker='o')

    plt.show()


# --------------------End Of Round 1-------------------------------------------------------------------
