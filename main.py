import cv2
import numpy as np


img = cv2.imread('./road2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

low_t = 50
high_t = 150
edges = cv2.Canny(blur, low_t,high_t)

cv2.imshow('blur', blur)
cv2.imshow('gray', gray)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


edges = cv2.Canny(blur, low_t,high_t)
vertices = np.array([
    [(0,img.shape[0]),(450, 310),(490,310),(img.shape[1],img.shape[0])]
], dtype=np.int32)
mask = np.zeros_like(edges)
ignore_mask_color = (255,255,255)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)
cv2.imshow('masked_edges', masked_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []

    y_bottom = 540
    y_upper = 315

    for line in lines:
        for x1, y1, x2, y2 in line:
            # test and filter values to slope
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('int')  # Avoid division by zero
            if abs(slope) > 0.5 and abs(slope) < 0.8:
                b = y1 - slope * x1
                if slope > 0:
                    x_bottom_pos.append(int((y_bottom - b) / slope))
                    x_upper_pos.append(int((y_upper - b) / slope))
                else:
                    x_bottom_neg.append(int((y_bottom - b) / slope))
                    x_upper_neg.append(int((y_upper - b) / slope))

    # Calculate mean values
    mean_x_bottom_pos = int(np.mean(x_bottom_pos)) if x_bottom_pos else 0
    mean_x_upper_pos = int(np.mean(x_upper_pos)) if x_upper_pos else 0
    mean_x_bottom_neg = int(np.mean(x_bottom_neg)) if x_bottom_neg else 0
    mean_x_upper_neg = int(np.mean(x_upper_neg)) if x_upper_neg else 0

    # a new 2d array with means
    lines_mean = np.array([
        [mean_x_bottom_pos, int(np.mean(y_bottom)), mean_x_upper_pos, int(np.mean(y_upper))],
        [mean_x_bottom_neg, int(np.mean(y_bottom)), mean_x_upper_neg, int(np.mean(y_upper))]
    ])

    # Draw the lines
    for i in range(len(lines_mean)):
        cv2.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)

def draw_linesq(img, lines, color=[0, 0, 255], thickness=1):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

rho = 3
theta = np.pi / 180
threshold = 15
min_line_length = 150
max_line_gap = 60
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

draw_linesq(img, lines)

cv2.imshow('image_with_lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def draw_liness(img, lines, color=[0, 0, 255], thickness=5):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    vertices = np.array([[(0, frame.shape[0]), (450, 310), (490, 310), (frame.shape[1], frame.shape[0])]], dtype=np.int32)
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 10
    max_line_gap = 20
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)
    draw_lines(frame, lines)
    return frame

video_capture = cv2.VideoCapture('./road.mp4')

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:
        processed_frame = process_image(frame)
        cv2.imshow('Video with Lines', processed_frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    else:
        break


video_capture.release()
cv2.destroyAllWindows()