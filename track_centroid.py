############ INFO ############
# Author : Gyu-Hwan Lee (Korea Institute of Science and Technology, Seoul National University)
# Contact: gh.lee@kist.re.kr
# Written : 2023-02-28
# Last edit : 2024-03-02 (Gyu-Hwan Lee)
# Description : Code for tracking mouse body centroid given that mouse body is the only dark object in the visual area.
#               For your specific use, you might alter following parameters:
#                   MINIMUM_AREA_SIZE: minimum size of dark patch you would consider for detection (# of pixels)
#                   DILATE_KERNEL_SIZE, DILATE_ITER: how much smoothing you will apply to detected dark patches
#                   UPPER_THRESH, LOWER_THRESH: related to how bright the mouse body color is. Range: (0,0,0) ~ (255,255,255)
##############################

import os, joblib
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

###### PARAMETERS ###### 
MINIMUM_AREA_SIZE = 1500
DILATE_KERNEL_SIZE = 3
DILATE_ITER = 20
UPPER_THRESH = (65,65,65)
LOWER_THRESH = (45,45,45)

VIDEO_FPS = 40.0
START_FRAME = 1 # 1: from the start
END_FRAME = -1 # -1: to the end
######################## 

DATA_DIR = os.path.join("..", "Data")
RESULT_DIR = os.path.join("..", "Results")
OUTPUT_DIR = "centroid_tracking"

out_path = os.path.join(RESULT_DIR, OUTPUT_DIR)
if not os.path.exists(out_path):
    os.mkdir(out_path)

def set_roi(img, description, roi_name="ROI"):
    print(f"  [INFO] Starting setting {roi_name} from the frame")
    img_height = img.shape[0]
    if img_height > 500:
        font_size = 1
        text_thickness = 2
        circle_size1 = 5
        circle_size2 = 10
        line_width = 4
    else:
        font_size = 0.5
        text_thickness = 1
        circle_size1 = 3
        circle_size2 = 6
        line_width = 2

    pts = []
    # mouse callback function
    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            pts.pop()

        if len(pts) > 0:
            # Draw the last point in pts
            cv2.circle(img2, pts[-1], circle_size1, (0, 0, 255), -1)

        if len(pts) > 1:
            for i in range(len(pts) - 1):
                # x ,y is the coordinates of the mouse click place
                cv2.circle(img2, pts[i], circle_size2, (0, 0, 255), -1)
                cv2.line(img2, pt1=pts[i], pt2=pts[i + 1],
                        color=(255, 0, 0), thickness=line_width)

        cv2.imshow('image', img2)

    # Create images and windows and bind windows to callback functions
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', draw_roi, img)
    cv2.putText(img, f"Determine your {roi_name} in the video", (round(img.shape[1]*0.8)//2, img.shape[0]//2), \
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), text_thickness)
    cv2.putText(img, description, (round(img.shape[1]*0.8)//2, round(img.shape[0]*1.2)//2), \
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255), text_thickness)
    print(f"  [INFO] Determine your {roi_name} in the video")
    print("  [INFO] Left click: select the point, right click: delete the last selected point")
    print("  [INFO] Press 'S' to determine the selection area and save it")
    print("  [INFO] Press ESC to quit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s") or key == ord("S"):
            roi = pts
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    
    return np.array(roi)

def crop_roi(img, points):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)

    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def within_area(point, roi):
    x, y = point
    rect = cv2.boundingRect(roi) # (x,y,w,h)
    
    return (rect[0] <= x <= (rect[0]+rect[2])) and (rect[1] <= y <= (rect[1]+rect[3]))

video_files = glob(os.path.join(DATA_DIR, "*_vid.avi"))
trials = [os.path.basename(video_file).split("_vid")[0] for video_file in video_files] 
print(f"[INFO] Found trials: {trials}\n")

for trial_idx, trial in enumerate(trials):
    video_file = video_files[trial_idx]
    print(f"[INFO] Start processing trial: {trial}")

    track_roi_file = os.path.join(DATA_DIR, f"{trial}_track_roi.pkl")

    ## Determine ROIs
    print("[INFO] Setting up ROIs")

    # 1. Track ROI
    if os.path.isfile(track_roi_file):
        print("  [INFO] Using previously saved track ROI information")
        track_roi = joblib.load(track_roi_file)

    else:
        vid = cv2.VideoCapture(video_file)
        # get the first frame
        ret, img = vid.read()
        vid.release()
        # set up the roi interactively
        track_roi = set_roi(img.copy(), "Arena: space within which animals can move around", 
                            roi_name="Arena")
        print(f"  Determined ROI: {track_roi}\n")
        # save roi points (as pkl) and the setting (as image)
        joblib.dump(value = track_roi, filename = track_roi_file)

    # prepare tracking
    vid = cv2.VideoCapture(video_file)
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    kernel = np.ones((DILATE_KERNEL_SIZE,DILATE_KERNEL_SIZE),np.uint8)
    out_file = os.path.join(out_path, f"{trial}_tracking_result.txt")

    if START_FRAME != 1:
        vid.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME-1)
        frame_idx = START_FRAME
    else:  
        frame_idx = 1

    # prepare for writing a video that shows tracking results
    _,_,w,h = cv2.boundingRect(track_roi) # (x,y,w,h)
    save_video_size = (2*w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(out_path, f"{trial}_tracking_video.mp4"), fourcc, VIDEO_FPS, save_video_size)

    of = open(out_file, 'w')
    print("Frame\tPos_XY", file=of)

    # Start tracking
    for idx in tqdm(range(n_frames)):
        if frame_idx > n_frames:
            break

        if END_FRAME > -1:
            if frame_idx == END_FRAME:
                break

        check, frm = vid.read()
        frm = crop_roi(frm, track_roi)
        
        # to black-white image
        frm_bw = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        frm_bw = cv2.cvtColor(frm_bw, cv2.COLOR_GRAY2RGB)
        
        mask = cv2.inRange(frm_bw, LOWER_THRESH, UPPER_THRESH)
        
        # erosion and dilation
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=DILATE_ITER)
        mask = cv2.erode(mask, kernel, iterations=DILATE_ITER-2)
        
        # get connected components
        ret, labels = cv2.connectedComponents(np.uint8(mask))

        # connectedComponents return the result array with 0 in background pixels.
        # skip 0 during the loop for 'i'
        cc_info = []
        for i in range(ret):
            if i == 0:
                continue
            # x, y : x- and y-indices where the array value is non-zero
            # len(x) == len(y) -> # of nonzero pixels
            y, x = (labels==i).nonzero()
            cc_size = len(x)
            
            if cc_size < MINIMUM_AREA_SIZE:
                # too big or too small
                continue

            # find the center of island
            center_x = int(x.mean())
            center_y = int(y.mean())

            cc_info.append([center_y, center_x, cc_size])
        
        # find the largest connected component
        pos = (-1,-1)
        cc_size = 0
        for y, x, curr_size in cc_info:
            if curr_size > cc_size:
                pos = (x, y)
                cc_size = curr_size
        
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Annotation & Visualization
        for y, x, _ in cc_info:
            frm_bw = cv2.circle(frm_bw, (x, y), 5, (0,0,255), -1) 
        
        for y, x, _ in cc_info:
            mask_3ch = cv2.circle(mask_3ch, (x, y), 5, (0,0,255), -1) 
        
        concat = cv2.hconcat([frm_bw, mask_3ch])
        out.write(concat)
        print(f"{frame_idx}\t{pos}", file=of)
        
        frame_idx += 1

    # close all open objects
    vid.release()
    out.release()
    of.close()
