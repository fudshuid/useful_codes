############ INFO ############
# Author : Gyu-Hwan Lee (Korea Institute of Science and Technology, Seoul National University)
# Contact: gh.lee@kist.re.kr
# Written : 2021-10-31
# Last edit : 2023-12-14 (Gyu-Hwan Lee)
# Description : Code for tracking object(s) with particular color in a video. Especially designed for videos where light condition is not uniform in time and/or space.
##############################

import os, cv2, joblib, time, gc, argparse
from glob import glob

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy.stats import circmean

###### PARAMS ######
# Blur on RGB frame: before superpixel segmentation
FRAME_BLUR_KERNEL_RGB = (0, 0)
FRAME_BLUR_SIGMA_RGB = 1

# Blur on HSV frame: for color space computation
FRAME_BLUR_KERNEL_HSV = (3, 3)
FRAME_BLUR_SIGMA_HSV = 0

COLORSPACE_SEARCH_JUMP_FRAMES = 5
INITIAL_HUE_SHIFT = 0 # 15

COLORSPACE_RES_DICT = {'H': 6, 'S': 8, 'V':8}
COLORSPACE_CUTS_DICT = {'H': np.arange(0, 181, COLORSPACE_RES_DICT['H']),
                        'S': np.arange(0, 257, COLORSPACE_RES_DICT['S']),
                        'V': np.arange(0, 257, COLORSPACE_RES_DICT['V'])}

COLORSPACE_ERODE_DILATE_PLAN = ['d/1', 'e/1', 'd/4']
COLORSPACE_ERODE_DILATE_PLAN_INITIAL = ['d/2', 'e/1', 'd/4']

DETECTED_PXLS_ERODE_DILATE_PLAN = ['e/4', 'd/7', 'e/3']
DETECTED_PXLS_KERNEL_SIZE = 3
MIN_CC_SIZE = 100

## Some combinations
# COLOR_RATIO_THRESH 0.15 & DISTANCE_THRESH 100
COLOR_RATIO_THRESH = 0.1

COLORSPACE_UPDATE_TERM = 2
COLORSPACE_EXPAND_THRESH = 2
COLORSPACE_SHRINK_THRESH = 0.6
COLORSPACE_RETAIN_THRESH_FWD = 0.1
COLORSPACE_RETAIN_THRESH_BWD = 0.1

HUE_SPREAD_LIMIT = 30
MIN_SATURATION = 80
JUMP_THRESH = 80 #50
DISTANCE_THRESH = JUMP_THRESH*2

FELZ_SCALE = 30
FELZ_MIN_SIZE = 50

FIGSIZE_FRAME = (11, 5)
FIGSIZE_COLORSPACE = (4, 4)

PROGRESS_REPORT_TERM = 50
SAVE_STATE_TERM = 200
VISUALIZE_TERM = 50

ap = argparse.ArgumentParser()
ap.add_argument("-x", "--ext", type=str, default="avi",
    help="which video extension should be searched (avi, mp4 etc.)")
ap.add_argument("-rp", "--readpath", type=str, default=os.path.join("..", "Data"),
    help="video path; directory where videos to analyze are found")
ap.add_argument("-sp", "--savepath", type=str, default=os.path.join("..", "Results"),
    help="result path; directory where extracted timepoints should be saved")
ap.add_argument("-sf", "--start_frame", type=int, default=1,
    help="first frame to synthesize")
ap.add_argument("-vf", "--video_file", type=str,
    help="name of the video file to perform color tracking")
ap.add_argument("--overwrite", action='store_true',
    help="whether to ignore existing tracking results")
ap.add_argument("--visualize", action='store_true',
    help="whether to visualize frame tracking results")
ap.add_argument("--debug", action='store_true',
    help="whether to show time to complete each step (for debugging/analysis)")
args = ap.parse_args()

DATA_DIR = args.readpath
RESULT_DIR = args.savepath

START_FRAME = args.start_frame
OVERWRITE = args.overwrite
VISUALIZE = args.visualize
SHOW_TIME_SPENT = args.debug
#####################

##### FUNCTIONS #####
def set_roi(img, description, roi_name="ROI"):
    print(f"[INFO] Starting setting {roi_name} from the frame")
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
            pts.append([x, y])

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
    print(f"[INFO] Determine your {roi_name} in the video")
    print("[INFO] Left click: select the point, right click: delete the last selected point")
    print("[INFO] Press 'S' or 's' to determine the selection area and save it")
    print("[INFO] Press 'P' or 'p' to pass this frame and see another frame")
    print("[INFO] Press ESC to quit")

    is_roi_set = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s") or key == ord("S"):
            roi = pts
            is_roi_set = True
            break
        if key == ord("p") or key == ord("P"):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    
    if is_roi_set: 
        return is_roi_set, np.array(roi)
    else:
        return is_roi_set, np.array([])

def get_start_position(img):
    img_height = img.shape[0]
    if img_height > 500:
        circle_size = 10
    else:
        circle_size = 6

    pts = []
    # mouse callback function
    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            pts.append([y, x])

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            pts.pop()

        if len(pts) > 0:
            for y, x in pts:
                # Draw the last point in pts
                cv2.circle(img2, [x, y], circle_size, (0, 0, 255), -1)

        cv2.imshow('image', img2)

    # Create images and windows and bind windows to callback functions
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', draw_roi, img)

    print("    [INFO] Determine the start position of the subject in the video")
    print("    [INFO] Left click: select the point, right click: delete the last selected point")
    print("    [INFO] Press 'S' or 's' to determine the selection area and save it")
    print("    [INFO] Press ESC to quit\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s") or key == ord("S"):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)

    return pts[0]

def crop_frame(img, roi):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)

    cv2.drawContours(mask, [roi], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(roi)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def annotate_segment(img, segment_mtrx, segment_idx):
    color = (0,0,0)
    color_rect = np.tile(color, (img.shape[0], img.shape[1], 1)).astype('float32')
    colored = cv2.addWeighted(img, 0.5, color_rect, 0.5, 0)
    
    idxs_of_segment = np.where(segment_mtrx == segment_idx)
    
    for xidx, yidx in zip(idxs_of_segment[0], idxs_of_segment[1]):
        img[xidx, yidx] = colored[xidx, yidx]
    
    return img

def select_segments(img, segment_mtrx, verbose=True):
    segment_idxs = []
    # mouse callback function
    def pick_segment(event, x, y, flags, param):
        img2 = img.copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            selected_segment_idx = segment_mtrx[y, x]
            segment_idxs.append(selected_segment_idx)

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            segment_idxs.pop()

        if len(segment_idxs) > 0:
            # Draw the last point in pts
            for segment_idx in segment_idxs:
                img2 = annotate_segment(img2, segment_mtrx, segment_idx)

        cv2.imshow('Image', img2)

    # Create images and windows and bind windows to callback functions
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', pick_segment, img)

    if verbose:
        print("    [INFO] Pick the segments with the color of the object you want to track")
        print("    [INFO] Left click: select the segment, right click: delete the last selected segment")
        print("    [INFO] Press 'S' to determine the selection and save it")
        print("    [INFO] If the subject doesn't exist in the displayed frame, press ESC to jump to future frames")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            segment_idxs = []
            break
        if key == ord("s") or key == ord("S"):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    
    return segment_idxs

def euc_dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def get_centroid(detection_img, current_pos, plan, kernel_size, min_cc_size, jump_thresh):
    # get connected components
    ret, labels = cv2.connectedComponents(detection_img)

    for i in range(ret):
        if i == 0:
            continue
        # x, y : x- and y-indices where the array value is non-zero
        # len(x) == len(y) -> # of nonzero pixels
        y, x = (labels==i).nonzero()

        # find the center of the connected component
        center_x = int(x.mean())
        center_y = int(y.mean())

        if current_pos != [-1, -1]:
            dist = euc_dist(current_pos, [center_y, center_x])
            if dist > jump_thresh:
                for ypos, xpos in zip(y, x):
                    detection_img[ypos, xpos] = 0
    
    kernel = np.ones([kernel_size, kernel_size], np.uint8)

    for instruction in plan:
        operation = instruction.split('/')[0]
        n_iter = int(instruction.split('/')[1])

        if operation == 'd':
            detection_img = cv2.dilate(detection_img, kernel, iterations=n_iter)
        elif operation == 'e':
            detection_img = cv2.erode(detection_img, kernel, iterations=n_iter)

    # get connected components
    ret, labels = cv2.connectedComponents(detection_img)

    # connectedComponents return the result array with 0 in background pixels.
    # skip 0 during the loop for 'i'
    max_size = 0
    centroid = [-1, -1]
    for i in range(ret):
        if i == 0:
            continue
        # x, y : x- and y-indices where the array value is non-zero
        # len(x) == len(y) -> # of nonzero pixels
        y, x = (labels==i).nonzero()
        cc_size = len(x)

        # find the center of the connected component
        center_x = int(x.mean())
        center_y = int(y.mean())

        if cc_size > max_size:
            max_size = cc_size
            centroid = [center_y, center_x]

    return centroid, detection_img

def convert_hsv_to_rgb(hsv):
    img_hsv = np.tile(hsv, (2, 2, 1)).astype('uint8')
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return list(img_rgb[0,0,:])

def convert_hsv_to_rgb_batch(colors_hsv):
    img_hsv = np.expand_dims(np.stack(colors_hsv, axis=0), axis=1).astype('uint8')
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return np.squeeze(img_rgb)

def make_colorspace_matrix(color_list, res_dict):
    axis_size_dict = {'H': 180, 'S': 256, 'V': 256}
    dims = [int(axis_size_dict[dim]/res_dict[dim]) for dim in ['H', 'S', 'V']]
    mat = np.zeros(dims).astype(int)

    for h, s, v in color_list:
        hidx = h // res_dict['H']
        sidx = s // res_dict['S']
        vidx = v // res_dict['V']
        mat[hidx, sidx, vidx] = 1
    
    return mat

def dense_to_sparse_idx(dense_idx, res_dict, dim):
    axes = ['H', 'S', 'V']
    res = res_dict[axes[dim]]

    return dense_idx // res

def sparse_to_dense_idx(sparse_idx, cuts_dict, dim):
    axes = ['H', 'S', 'V']
    cuts = cuts_dict[axes[dim]]

    return int((cuts[sparse_idx] + cuts[sparse_idx+1])/2)

def hue_circ_mean(hvals, n_pts):
    step_angle = 2 * np.pi / n_pts
    hval_rads = [step_angle * h for h in hvals]
    mean_hval_rad = circmean(hval_rads)
    mean_hval = round(mean_hval_rad / step_angle)
    
    return mean_hval

def get_colorspace_center(colorspace, cuts_dict):
    color_idxs = np.where(colorspace == 1)
    mean_color = [hue_circ_mean(color_idxs[0], colorspace.shape[0]), round(np.mean(color_idxs[1])), round(np.mean(color_idxs[2]))]
    
    return [sparse_to_dense_idx(v, cuts_dict, i) for i, v in enumerate(mean_color)]

def trim_colorspace(colorspace, color_range, minimum_saturation, res_dict):
    color_range_sparse = [dense_to_sparse_idx(v, res_dict, 0) for v in color_range]
    minimum_saturation_sparse = dense_to_sparse_idx(minimum_saturation, res_dict, 1)
    hdim = colorspace.shape[0]
    
    if color_range_sparse[0] < 0 and color_range_sparse[1] < hdim:
        colorspace[color_range_sparse[1]+1:hdim+color_range_sparse[0], :, :] = 0
    elif color_range_sparse[0] >= 0 and color_range_sparse[1] >= hdim:
        colorspace[color_range_sparse[1]-hdim+2:color_range_sparse[0], :, :] = 0
    elif color_range_sparse[0] >= 0 and color_range_sparse[1] < hdim:
        colorspace[:color_range_sparse[0], :, :] = 0
        colorspace[color_range_sparse[1]+1:, :, :] = 0
        
    colorspace[:, :minimum_saturation_sparse, :] = 0

    return colorspace

def dilate_colorspace_matrix(colorspace):
    color_idxs = np.where(colorspace == 1)
    for idx1, idx2, idx3 in zip(color_idxs[0], color_idxs[1], color_idxs[2]):
        idx2_low = max(idx2-1, 0)
        idx2_high = min(idx2+1, colorspace.shape[1])
        idx3_low = max(idx3-1, 0)
        idx3_high = min(idx3+1, colorspace.shape[2])
        
        if idx1 == 0:
            colorspace[:2, idx2_low:idx2_high+1, idx3_low:idx3_high+1] = 1
            colorspace[colorspace.shape[0]-1, idx2_low:idx2_high+1, idx3_low:idx3_high+1] = 1
        elif idx1 == colorspace.shape[0]-1:
            colorspace[0, idx2_low:idx2_high+1, idx3_low:idx3_high+1] = 1
            colorspace[colorspace.shape[0]-2:, idx2_low:idx2_high+1, idx3_low:idx3_high+1] = 1
        else:
            colorspace[idx1-1:idx1+2, idx2_low:idx2_high+1, idx3_low:idx3_high+1] = 1
        
    return colorspace

def erode_colorspace_matrix(colorspace):
    color_idxs = np.where(colorspace == 1)
    to_remove = []
    for idx1, idx2, idx3 in zip(color_idxs[0], color_idxs[1], color_idxs[2]):
        idx2_low = max(idx2-1, 0)
        idx2_high = min(idx2+1, colorspace.shape[1])
        idx3_low = max(idx3-1, 0)
        idx3_high = min(idx3+1, colorspace.shape[2])
        
        collect_neighbors = []
        if idx1 == 0:
            collect_neighbors.append(colorspace[:2, idx2_low:idx2_high+1, idx3_low:idx3_high+1])
            collect_neighbors.append(np.expand_dims(colorspace[colorspace.shape[0]-1, idx2_low:idx2_high+1, idx3_low:idx3_high+1], axis=0))
        elif idx1 == colorspace.shape[0]-1:
            collect_neighbors.append(np.expand_dims(colorspace[0, idx2_low:idx2_high+1, idx3_low:idx3_high+1], axis=0))
            collect_neighbors.append(colorspace[colorspace.shape[0]-2:, idx2_low:idx2_high+1, idx3_low:idx3_high+1])
        else:
            collect_neighbors.append(colorspace[idx1-1:idx1+2, idx2_low:idx2_high+1, idx3_low:idx3_high+1])
        
        neighbors = np.concatenate(collect_neighbors, axis=0)
        if np.sum(neighbors) < neighbors.size:
            to_remove.append([idx1, idx2, idx3])
    
    for idx1, idx2, idx3  in to_remove:
        colorspace[idx1, idx2, idx3] = 0
    
    return colorspace

def erode_and_dilate_colorspace(colorspace, plan):
    for instruction in plan:
        operation = instruction.split('/')[0]
        n_iter = int(instruction.split('/')[1])
                
        for i in range(n_iter):
            if operation == 'd':
                colorspace = dilate_colorspace_matrix(colorspace)
            elif operation == 'e':
                colorspace = erode_colorspace_matrix(colorspace)
        
    return colorspace

def combine_colorspaces(colorspaces):
    n_spaces = len(colorspaces)

    if n_spaces == 1:
        combined_colorspace = colorspaces[0]

    else:
        combined_colorspace = np.logical_or(colorspaces[0], colorspaces[1])
        for i in range(2, n_spaces):
            combined_colorspace = np.logical_or(combined_colorspace, colorspaces[i])

    return combined_colorspace

def within_colorspace(colorspace_matrix, hsv, res_dict):
    h_sparse, s_sparse, v_sparse = [dense_to_sparse_idx(idx, res_dict, d) for d, idx in enumerate(hsv)]

    return colorspace_matrix[h_sparse, s_sparse, v_sparse] == 1

def get_colorspace_retain_ratio(colorspace_prev, colorspace_new):
    color_idxs = np.where(colorspace_prev == 1)
    colorspace_size_prev = len(color_idxs[0])
    
    retained_count = 0
    for idx1, idx2, idx3 in zip(color_idxs[0], color_idxs[1], color_idxs[2]):
        if colorspace_new[idx1, idx2, idx3] == 1:
            retained_count += 1
    
    return retained_count / colorspace_size_prev

def visualize_colorspace_hsv(colorspace, ax, cuts_dict):
    colors_hsv = []
    color_idxs = np.where(colorspace == 1)
    for hsv in zip(color_idxs[0], color_idxs[1], color_idxs[2]):
        colors_hsv.append([sparse_to_dense_idx(idx, cuts_dict, d) for d, idx in enumerate(hsv)])
    
    pixel_colors = convert_hsv_to_rgb_batch(colors_hsv)

    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    
    colors_hsv = np.array(colors_hsv)
    h = colors_hsv[:,0]
    s = colors_hsv[:,1]
    v = colors_hsv[:,2]

    ax.view_init(elev=45, azim=60, roll=0)
    ax.set_box_aspect(aspect=(1,1,1), zoom=0.8)

    ax.scatter(h, s, v, facecolors=pixel_colors, marker=".", alpha=0.5, s=30)
    ax.set_xlabel("Hue", fontsize=12, labelpad=10)
    ax.set_ylabel("Saturation", fontsize=12, labelpad=10)
    ax.set_zlabel("Value", fontsize=12, labelpad=10)
    
    ax.set_xlim([0, 180])
    ax.set_ylim([0, 256])
    ax.set_zlim([0, 256])
    
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_yticks(np.arange(0, 257, 64))
    ax.set_zticks(np.arange(0, 257, 64))

#####################

if args.video_file is not None:
    print(f"[INFO] Video file was input as an argument: {args.video_file}")
    video_file = args.video_file

else:
    video_files = [os.path.basename(path) for path in sorted(glob(os.path.join(DATA_DIR, f"*.{args.ext}")))]

    print(f"[INFO] Detected {len(video_files)} videos:")
    for idx, video_path in enumerate(video_files):
        print(f"{idx+1}. {video_path}")
    print()

    print(f"[INFO] Type the number of video (index range: 1 ~ {len(video_files)}) to perform color tracking")
    choice_idx = int(input("[PROMPT] Enter your choice: "))
    video_file = video_files[choice_idx-1]

print(f"[INFO] Video to track: {video_file}")
video_path = os.path.join(DATA_DIR, video_file)
video_name = os.path.basename(video_file).split(".")[0]

out_path = os.path.join(RESULT_DIR, "color_tracking")
if not os.path.exists(out_path):
    os.mkdir(out_path)
    
video_frames_path = os.path.join(out_path, video_name)
if not os.path.exists(video_frames_path):
    os.mkdir(video_frames_path)

saved_state_file = os.path.join(out_path, f"{video_name}_state_dict.pkl")
initial_saved_state_file = os.path.join(out_path, f"{video_name}_initial_state_dict.pkl")

# Set track ROI
vidcap = cv2.VideoCapture(video_path)
n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

roi_path = os.path.join(DATA_DIR, f"{os.path.basename(video_file).split('.')[0]}_color_track_roi.pkl")
if os.path.isfile(roi_path):
    print("[INFO] Saved LED ROI file exists for this video. Loading it ...")
    roi = joblib.load(roi_path)
else:
    print("[INFO] Setting up LED ROI for this video ...")
    randomized_idx = np.random.permutation(n_frames)
    is_roi_set = False
    idx_position = 0
    
    while not is_roi_set:
        frame_idx = randomized_idx[idx_position]
        idx_position += 1

        if frame_idx > 1:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-1)

        _, frame = vidcap.read()
        is_roi_set, roi = set_roi(frame, "Draw a rectangle around the Arena", roi_name="Color Track ROI")

    joblib.dump(roi, roi_path)

if os.path.isfile(saved_state_file) and not OVERWRITE:
    print("\n[INFO] Saved state file exists for this video. Loading it ...")
    state_dict = joblib.load(saved_state_file)

    subjects = state_dict['subjects']    
    current_frame = state_dict['current_frame']
    frame_count = state_dict['frame_count']

    result_txt_file = os.path.join(DATA_DIR, f"{video_name}_tracking_result_{subjects}.txt")

    # check the last character
    end_with_newline = False
    with open(result_txt_file, 'r') as rf:
        content = rf.read()
        last_char = content[-1]
        if last_char == '\n':
            end_with_newline = True

    out_file = open(result_txt_file, 'a')
    if not end_with_newline:
        out_file.write("\n")

    print(f"[INFO] Tracking will resume from frame #{current_frame}")

    enter_frame_dict = {}
    colorspace_matrix_dict = {}
    colorspace_size_dict = {}
    current_pos_dict = {}
    backup_pos_dict = {}
    center_color_dict = {}
    hue_range_dict = {}
    
    for subject in subjects:
        enter_frame_dict[subject] = state_dict[subject]['enter_frame']
        colorspace_matrix_dict[subject] = state_dict[subject]['colorspace']
        colorspace_size_dict[subject] = state_dict[subject]['colorspace_size']
        current_pos_dict[subject] = state_dict[subject]['current_pos']
        backup_pos_dict[subject] = state_dict[subject]['backup_pos']
        center_color_dict[subject] = state_dict[subject]['center_color']
        hue_range_dict[subject] = state_dict[subject]['hue_range']

else:
    # construct initial color space
    n_subjects = int(input("\n[PROMPT] How many subjects do you want to track? Type a number: "))
    
    subjects = []
    start_position_dict = {}
    enter_frame_dict = {}
    colorspace_matrix_dict = {}
    colorspace_size_dict = {}
    current_pos_dict = {}
    backup_pos_dict = {}
    center_color_dict = {}
    hue_range_dict = {}

    for i in range(n_subjects):
        subject = input(f"\n  [PROMPT] How should Subject #{i+1} be called? Type in a string: ")
        subjects.append(subject)

        print(f"  [PROMPT] (Optional) When does subject '{subject}' enter?")
        print("  [PROMPT] If you don't want to specify, just press ENTER.")
        input_str = input(f"  [PROMPT] Otherwise, type a number or an expression (eg. 100, 10*10): ")
        if len(input_str) == 0:
            enter_frame = START_FRAME
        else:
            enter_frame = eval(input_str)

        enter_frame_dict[subject] = enter_frame

    result_txt_file = os.path.join(DATA_DIR, f"{video_name}_tracking_result_{subjects}.txt")

    out_file = open(result_txt_file, 'w')
    columns = ['Frame'] + [f"{subject}_PosXY" for subject in subjects]
    print("\t".join(columns), file=out_file)

    print(f"\n[INFO] Start receiving start positin of subjects")
    for subject in subjects:
        print(f"  [INFO] Subject: {subject}")
        frame_idx = enter_frame_dict[subject]

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-1)

        _, frame = vidcap.read()
        frame = crop_frame(frame, roi)

        start_position = get_start_position(frame)
        start_position_dict[subject] = start_position

    print(f"\n[INFO] Start constructing initial colorspaces")
    for subject in subjects:
        print(f"  [INFO] Subject: {subject}")
        segments_selected = False
        frame_idx = enter_frame_dict[subject]
        is_first_trial = True

        while not segments_selected:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-1)

            _, frame = vidcap.read()
            frame = crop_frame(frame, roi)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.GaussianBlur(frame_rgb, FRAME_BLUR_KERNEL_RGB, FRAME_BLUR_SIGMA_RGB)

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # frame_hsv = cv2.GaussianBlur(frame_hsv, FRAME_BLUR_KERNEL_HSV, FRAME_BLUR_SIGMA_HSV)

            img = img_as_float(frame_rgb)
            segments = felzenszwalb(img, scale=FELZ_SCALE//2, sigma=0, min_size=FELZ_MIN_SIZE//2)
            img_with_boundaries = mark_boundaries(img, segments, color=(1,1,1))

            img_annotated = cv2.addWeighted(img, 0.7, img_with_boundaries, 0.3, 0).astype('float32')
            img_annotated_bgr = cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR)

            selected_segment_idxs = select_segments(img_annotated_bgr, segments, is_first_trial)

            if len(selected_segment_idxs) > 0:
                segments_selected = True
            
            frame_idx += COLORSPACE_SEARCH_JUMP_FRAMES
            if is_first_trial:
                is_first_trial = False

        selected_colors_hsv = []
        for segment_idx in selected_segment_idxs:
            selected_colors_hsv += frame_hsv[segments == segment_idx].tolist()

        # shift hue value
        selected_colors_hsv = [[min(h+INITIAL_HUE_SHIFT, 179),s,v] for h,s,v in selected_colors_hsv]

        colorspace_matrix = make_colorspace_matrix(selected_colors_hsv, COLORSPACE_RES_DICT)
        colorspace_matrix = erode_and_dilate_colorspace(colorspace_matrix, COLORSPACE_ERODE_DILATE_PLAN_INITIAL)
        colorspace_center = get_colorspace_center(colorspace_matrix, COLORSPACE_CUTS_DICT)
        hue_center = colorspace_center[0]
        hue_range = [max(0, hue_center - HUE_SPREAD_LIMIT), min(hue_center + HUE_SPREAD_LIMIT, 179)]
        colorspace_matrix = trim_colorspace(colorspace_matrix, hue_range, MIN_SATURATION, COLORSPACE_RES_DICT)
        colorspace_size = np.sum(colorspace_matrix)

        # save calculated values to current state dictionaries
        colorspace_matrix_dict[subject] = colorspace_matrix
        colorspace_size_dict[subject] = colorspace_size
        current_pos_dict[subject] = [-1, -1]
        backup_pos_dict[subject] = [-1, -1]
        center_color_dict[subject] = colorspace_center
        hue_range_dict[subject] = hue_range
    
    vidcap.release()

    frame_count = 1
    current_frame = START_FRAME

    print("[PROGRESS] Saving initial state dict ...")
    state_dict = {'subjects': subjects, 'current_frame': current_frame, 'frame_count': frame_count}

    for subject in subjects:
        state_dict[subject] = {}

        state_dict[subject]['enter_frame'] = enter_frame_dict[subject]
        state_dict[subject]['colorspace'] = colorspace_matrix_dict[subject]
        state_dict[subject]['colorspace_size'] = colorspace_size_dict[subject]
        state_dict[subject]['current_pos'] = current_pos_dict[subject]
        state_dict[subject]['backup_pos'] = backup_pos_dict[subject]
        state_dict[subject]['center_color'] = center_color_dict[subject]
        state_dict[subject]['hue_range'] = hue_range_dict[subject]

    joblib.dump(state_dict, initial_saved_state_file)

# start tracking
print(f"\n[INFO] Start tracking ...")

vidcap = cv2.VideoCapture(video_path)
n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_frame-1)

fig_frame = plt.figure(figsize=FIGSIZE_FRAME)
fig_colorspace = plt.figure(figsize=FIGSIZE_COLORSPACE)

saved_time = time.time()
visualized_frame_count = 1
while current_frame <= n_frames:
    if frame_count % PROGRESS_REPORT_TERM == 0:
        curr_time = time.time()
        elapsed_time = curr_time - saved_time
        print(f"  [PROGRESS] Tracking Frame #{current_frame} out of {n_frames} frames (Speed: {elapsed_time/PROGRESS_REPORT_TERM:.3f}s / Frame)")
        saved_time = curr_time

    if SHOW_TIME_SPENT:
        print(f"Processing frame #{current_frame}:")

    ret, frame = vidcap.read()
    frame = crop_frame(frame, roi)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.GaussianBlur(frame_rgb, FRAME_BLUR_KERNEL_RGB, FRAME_BLUR_SIGMA_RGB)
    
    # for visualization
    frm_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame_hsv = cv2.GaussianBlur(frame_hsv, FRAME_BLUR_KERNEL_HSV, FRAME_BLUR_SIGMA_HSV)

    segments_matrices = {}
    for subject in subjects:
        enter_frame = enter_frame_dict[subject]

        if current_frame < enter_frame:
            continue

        colorspace_matrix = colorspace_matrix_dict[subject]
        colorspace_size = colorspace_size_dict[subject]
        if current_frame == enter_frame:
            current_pos = start_position_dict[subject]
            backup_pos = start_position_dict[subject]
        else:
            current_pos = current_pos_dict[subject]
            backup_pos = backup_pos_dict[subject]
        hue_range = hue_range_dict[subject]

        detections = np.zeros(frame.shape[:-1], dtype=np.uint8)

        subject_color_hsv = center_color_dict[subject]
        subject_color_hsv[1] = 200
        subject_color_hsv[2] = 200
        subject_color_rgb = convert_hsv_to_rgb(subject_color_hsv)

        ### Step 1: Prepare and segment frame
        if SHOW_TIME_SPENT:
            start_time = time.time()

        img = img_as_float(frame_rgb)
        if current_pos[0] != -1 and backup_pos[0] != -1:
            # make reduced image
            top_lim = max(0, current_pos[0]-DISTANCE_THRESH)
            bottom_lim = min(img.shape[0], current_pos[0]+DISTANCE_THRESH)

            left_lim = max(0, current_pos[1]-DISTANCE_THRESH)
            right_lim = min(img.shape[1], current_pos[1]+DISTANCE_THRESH)

            img_to_segment = img[top_lim:bottom_lim, left_lim:right_lim, :]
            
        else:
            img_to_segment = img

        segments = felzenszwalb(img_to_segment, scale=FELZ_SCALE, sigma=0, min_size=FELZ_MIN_SIZE)
        
        segment_idxs = sorted(np.unique(segments.flatten()))
        n_segments = len(segment_idxs)

        if current_pos[0] != -1 and backup_pos[0] != -1:
            background = np.zeros(img.shape[:-1], dtype=int)
            background += segment_idxs[-1]+1
            background[top_lim:bottom_lim, left_lim:right_lim] = segments

            segments = background

        segments_matrices[subject] = segments

        if current_pos[0] == -1 and backup_pos[0] != -1:
            current_pos = backup_pos
        
        if SHOW_TIME_SPENT:
            end_time = time.time()
            print(f"  Step 1 took {end_time - start_time:.3f}s")

        ### Step 2: select segments based on within colorspace ratio
        if SHOW_TIME_SPENT:
            start_time = time.time()

        selected_segment_idxs = []
        for segment_idx in segment_idxs:
            idxs_in_segment = np.where(segments == segment_idx)
            n_pxls_in_segment = len(idxs_in_segment[0])
            
            segment_center = (np.round(np.mean(idxs_in_segment[0])), np.round(np.mean(idxs_in_segment[1])))
            
            if backup_pos[0] != -1:
                if euc_dist(segment_center, current_pos) > DISTANCE_THRESH:
                    continue
            
            n_pxls_in_colorspace = 0
            for idx1, idx2 in zip(idxs_in_segment[0], idxs_in_segment[1]):
                hsv = frame_hsv[idx1, idx2, :]
                if within_colorspace(colorspace_matrix, hsv, COLORSPACE_RES_DICT):
                    n_pxls_in_colorspace += 1
                    
            ratio_in_colorspace = n_pxls_in_colorspace/n_pxls_in_segment
            
            if ratio_in_colorspace > COLOR_RATIO_THRESH:
                selected_segment_idxs.append(segment_idx)

        if SHOW_TIME_SPENT:
            end_time = time.time()
            print(f"  Step 2 took {end_time - start_time:.3f}s")

        ### Step 3
        if len(selected_segment_idxs) > 0:
            # Step 3-1: update colorspace
            if SHOW_TIME_SPENT:
                start_time = time.time()
            
            if frame_count % COLORSPACE_UPDATE_TERM == 0:
                selected_colors_hsv = []
                for segment_idx in selected_segment_idxs:
                    selected_colors_hsv += frame_hsv[segments == segment_idx].tolist()

                colorspace_matrix_new = make_colorspace_matrix(selected_colors_hsv, COLORSPACE_RES_DICT)
                colorspace_matrix_new = erode_and_dilate_colorspace(colorspace_matrix_new, COLORSPACE_ERODE_DILATE_PLAN)
                colorspace_matrix_new = trim_colorspace(colorspace_matrix_new, hue_range, MIN_SATURATION, COLORSPACE_RES_DICT)
                colorspace_size_new = np.sum(colorspace_matrix_new)

                if (colorspace_size_new > COLORSPACE_SHRINK_THRESH*colorspace_size) and (colorspace_size_new < COLORSPACE_EXPAND_THRESH*colorspace_size):
                    colorspace_retain_ratio_fwd = get_colorspace_retain_ratio(colorspace_matrix, colorspace_matrix_new)
                    colorspace_retain_ratio_bwd = get_colorspace_retain_ratio(colorspace_matrix_new, colorspace_matrix)

                    if colorspace_retain_ratio_fwd > COLORSPACE_RETAIN_THRESH_FWD and colorspace_retain_ratio_bwd > COLORSPACE_RETAIN_THRESH_BWD:
                        colorspace_matrix_dict[subject] = colorspace_matrix_new
                        colorspace_size_dict[subject]= colorspace_size_new
            
            if SHOW_TIME_SPENT:
                end_time = time.time()
                print(f"  Step 3-1 took {end_time - start_time:.3f}s")

            # Step 3-2: find pixels within colorspace
            if SHOW_TIME_SPENT:
                start_time = time.time()
            
            pxls_selected = []
            for segment_idx in selected_segment_idxs:
                idxs_in_segment = np.where(segments == segment_idx)
                
                for idx1, idx2 in zip(idxs_in_segment[0], idxs_in_segment[1]):
                    detections[idx1, idx2] = 255
                    hsv = frame_hsv[idx1, idx2, :]
                    if within_colorspace(colorspace_matrix, hsv, COLORSPACE_RES_DICT):
                        pxls_selected.append([idx1, idx2])
            
            if SHOW_TIME_SPENT:
                end_time = time.time()
                print(f"  Step 3-2 took {end_time - start_time:.3f}s")

            # Step 3-3: get centroid update coordinates
            if SHOW_TIME_SPENT:
                start_time = time.time()

            centroid, detection_img = get_centroid(detections, current_pos, DETECTED_PXLS_ERODE_DILATE_PLAN, DETECTED_PXLS_KERNEL_SIZE, MIN_CC_SIZE, JUMP_THRESH)
            
            if VISUALIZE and (frame_count % VISUALIZE_TERM == 0):
                detected_pxl_idxs = np.where(detection_img == 255)
                for idx1, idx2 in zip(detected_pxl_idxs[0], detected_pxl_idxs[1]):
                    frm_rgb[idx1, idx2, :] = subject_color_rgb

            if current_pos != [-1, -1]:
                if euc_dist(centroid, current_pos) < JUMP_THRESH:
                    current_pos_dict[subject] = centroid
                    backup_pos_dict[subject] = centroid
            else:
                current_pos_dict[subject] = centroid
                backup_pos_dict[subject] = centroid

            if SHOW_TIME_SPENT:
                end_time = time.time()
                print(f"  Step 3-3 took {end_time - start_time:.3f}s")

        else:
            # this means the tracking is lost
            backup_pos_dict[subject] = current_pos
            current_pos_dict[subject] = [-1, -1]
            
    ### Step 4: write result & visualize
    if SHOW_TIME_SPENT:
        start_time = time.time()

    # write to result text file
    output_line_parts = [str(current_frame)] + [f"[{current_pos_dict[subject][1]}, {current_pos_dict[subject][0]}]" for subject in subjects]
    print("\t".join(output_line_parts), file=out_file)

    if VISUALIZE and (frame_count % VISUALIZE_TERM == 0) and any(current_frame >= enter_frame_dict[subject] for subject in subjects):
        # make composite frame
        save_path = os.path.join(video_frames_path, f"frame{visualized_frame_count:05}.jpg")
        fig_frame.suptitle(f"{' '.join(video_name.split('_')[:2])} | Frame #{current_frame:05}", y=0.98, fontsize=20)
        
        grid = fig_frame.add_gridspec(nrows=1, ncols=2, width_ratios=[3,2])
        ax1 = fig_frame.add_subplot(grid[0,0])
        ax2 = fig_frame.add_subplot(grid[0,1], projection="3d")    
        
        frm_rgb = img_as_float(frm_rgb)
        frm_with_boundaries = frm_rgb.copy()
        for subject in subjects:
            if subject in segments_matrices:
                segments = segments_matrices[subject]
                frm_with_boundaries = mark_boundaries(frm_with_boundaries, segments, color=(1,1,1))

        frm_annotated = cv2.addWeighted(frm_rgb, 0.8, frm_with_boundaries, 0.2, 0).astype('float32')

        for subject in subjects:
            current_pos = current_pos_dict[subject]
            if current_pos[0] != -1:
                frm_annotated = cv2.circle(frm_annotated, (current_pos[1], current_pos[0]), 5, (1, 1, 1), -1)
        
        ax1.imshow(frm_annotated)
        ax1.set_axis_off()
        ax1.set_title("Annotated Frame", fontsize=15, y=1.05)

        colorspace_matrices = [colorspace_matrix_dict[subject] for subject in subjects]
        visualize_colorspace_hsv(combine_colorspaces(colorspace_matrices), ax2, COLORSPACE_CUTS_DICT)
        ax2.set_title("Tracking colorspace", fontsize=15, y=0.99)
        
        fig_frame.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.1, dpi = 200)
        visualized_frame_count += 1

        plt.clf()
        plt.close(fig_frame)
        del grid, ax1, ax2
        del frm_annotated

    frame_count += 1
    current_frame += 1

    if SHOW_TIME_SPENT:
        end_time = time.time()
        print(f"  Step 4 took {end_time - start_time:.3f}s")

    ### Step 5: save state dict
    if SHOW_TIME_SPENT:
        start_time = time.time()

    if (frame_count % SAVE_STATE_TERM == 0) and any(current_frame >= enter_frame_dict[subject] for subject in subjects):
        print(f"  [PROGRESS] Saving state dict at Frame #{current_frame} ...")
        state_dict = {'subjects': subjects, 'current_frame': current_frame, 'frame_count': frame_count}

        for subject in subjects:
            state_dict[subject] = {}

            state_dict[subject]['enter_frame'] = enter_frame_dict[subject]
            state_dict[subject]['colorspace'] = colorspace_matrix_dict[subject]
            state_dict[subject]['colorspace_size'] = colorspace_size_dict[subject]
            state_dict[subject]['current_pos'] = current_pos_dict[subject]
            state_dict[subject]['backup_pos'] = backup_pos_dict[subject]
            state_dict[subject]['center_color'] = center_color_dict[subject]
            state_dict[subject]['hue_range'] = hue_range_dict[subject]
        
        joblib.dump(state_dict, saved_state_file)
        out_file.close()
        out_file = open(result_txt_file, 'a')
    
    gc.collect()

    if SHOW_TIME_SPENT:
        end_time = time.time()
        print(f"  Step 5 took {end_time - start_time:.3f}s")
        print()

vidcap.release()
