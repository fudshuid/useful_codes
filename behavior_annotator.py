############ INFO ############
# Author : Gyu-Hwan Lee (Korea Institute of Science and Technology, Seoul National University)
# Contact: gh.lee@kist.re.kr
# Written : 2021-10-31
# Last edit : 2023-12-14 (Gyu-Hwan Lee)
# Description : GUI code for behavioral annotation while looking at video. Provides many functionalities:
#                   saving video samples, skipping frames, adding/removing annotations, save annotation results at anytime into a text file
##############################

import cv2
import os, re, sys
import numpy as np
import joblib
from glob import glob
import matplotlib
# to ensure that figure_to_array() function works properly
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from pathlib import Path

DATA_DIR = os.path.join(".", "Source")
BEHAV_ANNOT_FILE = os.path.join(".", "behavior_annotations_interaction.txt")
VIDEO_SAVE_DIR = os.path.join(".", "SavedVideos")
if not os.path.exists(VIDEO_SAVE_DIR):
    os.makedirs(VIDEO_SAVE_DIR)

OVERWRITE = False
INTERVAL_S = 23
INTERVAL_L = 30
INTERVAL_S_SAVER = 25
INTERVAL_L_SAVER = 35
# the vertical length the frames are resized to
VSPAN = 800
MAX_FR = 200
FR_INTERVAL = 5
ANNOT_WINDOW_WIDTH = 230
VIDEO_SAVE_FPS = 20

FRAME_JUMP_SIZE = 500
REDUCE_SIZE_RATIO = 1

with open(BEHAV_ANNOT_FILE, 'r') as f:
    # skip the first line (the header)
    annotations = [line.strip().split('\t') for line in f.readlines()[1:]]

behaviors = [annotation for annotation, _ in annotations]

commands = [['SPACE', 'play/stop'],
            ['f/F', 'forward play'],
            ['b/B', 'backward play'],
            ['p/P', 'prev frame'],
            ['n/N', 'next frame'],
            ['j/J', 'jump frames'],
            ['k/K', 'play slower'],
            ['l/L', 'play faster'],
            ['s/S', 'save current'],
            ['v/V', 'save video'],
            ['ENTER', 'start/end annot'],
            ['ESC', 'quit']]

commands_dict = {ord(' '): 'play/stop', 
                ord('f'): 'forward', ord('F'): 'forward',
                ord('b'): 'backward', ord('B'): 'backward',
                ord('p'): 'prev_frame', ord('P'): 'prev_frame',
                ord('n'): 'next_frame', ord('N'): 'next_frame',
                ord('j'): 'jump_frame', ord('J'): 'jump_frame',
                ord('k'): 'slower', ord('K'): 'slower',
                ord('l'): 'faster', ord('L'): 'faster',
                ord('s'): 'save', ord('S'): 'save',
                ord('v'): 'video', ord('V'): 'video',
                27: 'exit', 13: 'annotate'}

def set_roi(img):
    # Create ROI
    pts = []
    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            pts.pop()

        if len(pts) > 0:
            # Draw the last point in pts
            cv2.circle(img2, pts[-1], 5, (0, 0, 255), -1)

        if len(pts) > 1:
            for i in range(len(pts) - 1):
                # x ,y is the coordinates of the mouse click place
                cv2.circle(img2, pts[i], 10, (0, 0, 255), -1)
                cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1],
                        color=(255, 0, 0), thickness=4)
        
        cv2.imshow('Setting ROI', img2)

    # Create images and windows and bind windows to callback functions
    cv2.imshow('Setting ROI', img)
    cv2.setMouseCallback('Setting ROI', draw_roi, img)
    print("[INFO] Determine ROI")
    print("[INFO] Left click: select the point, right click: delete the last selected point")
    print("[INFO] Press 'S' to determine the selection area and save it")
    print("[INFO] Press ESC to quit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s") or key == ord("S"):
            roi = pts
            break

    cv2.destroyAllWindows()
    for i in range(1, 5):
        cv2.waitKey(1)

    return np.array(roi)

def crop_roi(img, roi):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)

    cv2.drawContours(mask, [roi], -1, (255, 255, 255), -1, cv2.LINE_AA)
    res = cv2.bitwise_and(img, img, mask=mask)
    rect = cv2.boundingRect(roi)  # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def adjust_gamma(img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def parse_annotations(annotations, behaviors):
    # sort annotations with frame number
    annotations.sort(key = lambda x: x[0])
    
    # group annotations based on the behavior types
    annotation_dict = {}
    for behavior in behaviors:
        annotation_list = [a for a in annotations if a[1].split('/')[0] == behavior]
        annotation_dict[behavior] = annotation_list
    
    # rearrange annotations into list of epochs: [behavior, start, end]
    behavior_epochs = []
    for behavior in behaviors:
        annotation_list = [a for a in annotations if a[1].split('/')[0] == behavior]
        assert len(annotation_list)%2 == 0
        
        for idx in range(len(annotation_list)//2):
            assert (annotation_list[idx*2][1].split('/')[1] == 'S') and \
                    (annotation_list[idx*2+1][1].split('/')[1] == 'E')
            
            behavior_epochs.append([behavior, annotation_list[idx*2][0], annotation_list[idx*2+1][0]])
    
    return behavior_epochs

def figure_to_array(fig):
    # convert plt.figure into numpy array
    # shape: height, width, layer
    fig.canvas.draw()
    converted = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
    
    return converted

def get_timeline_img(parsed_behavs, behaviors, required_width):
    # make timeline image
    hspan = required_width

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(1, 1, 1)
    colors = [plt.cm.rainbow(val)
              for val in np.linspace(0.0, 1.0, len(behaviors))]

    ax.axhline(y=0.5, linewidth=4, color='k')

    for behav, start, end in parsed_behavs:
        ax.axvspan(xmin=start, xmax=end, ymin=0.25, ymax=0.75, 
                   color = colors[behaviors.index(behav)], alpha = 0.7)

    ax.set_xlim([1,n_frames])
    ax.set_xlabel("Frame #", fontsize=25, labelpad=15)
    ax.set_ylim([0,1])
    ax.set_yticklabels("")

    legend_lines = []
    for i, behav in enumerate(behaviors):
        legend_lines.append(Line2D([0], [0], color=colors[i], lw=10))

    ax.legend(legend_lines, behaviors, bbox_to_anchor=(0., 1.0, 1., .1),
              loc=3, ncol=8, mode="expand", borderaxespad=0.,
              prop={"size":20})

    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_minor_locator(MultipleLocator(1000))
    plt.xticks(fontsize=20, rotation=45)

    plt.grid(True, alpha=0.5, which='major')
    plt.grid(True, alpha=0.1, which='minor')
    plt.tight_layout()
    plt.close()
    
    arr = figure_to_array(fig)[:,:,0:3]

    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(arr, (hspan, int(arr.shape[0]*(hspan/arr.shape[1]))), interpolation=cv2.INTER_LINEAR)
    
    return resized

def remove_behav_epoch(behavior_annotations, parsed_behavs, pos_to_delete):
    # check which epoch is to delete
    idx_to_delete = -1
    for i, (behav, start, end) in enumerate(parsed_behavs):
        if start <= pos_to_delete <= end:
            idx_to_delete = i
            break
    
    # if no overlappig epoch was found
    if idx_to_delete == -1:
        return behavior_annotations, parsed_behavs
    
    # if that's not the case
    behav_to_del, start_to_del, end_to_del = parsed_behavs[idx_to_delete]
    
    # delete from parsed_behavs
    del(parsed_behavs[idx_to_delete])
    
    # delete from behavior_annotations
    idx = 0
    while idx <= len(behavior_annotations)-1:
        pos, annotation_str = behavior_annotations[idx]
        
        if (pos == start_to_del) and (annotation_str == '/'.join([behav_to_del, 'S'])):
            del(behavior_annotations[idx])
        elif (pos == end_to_del) and (annotation_str == '/'.join([behav_to_del, 'E'])):
            del(behavior_annotations[idx])
        else:
            idx += 1
    
    return behavior_annotations, parsed_behavs

def save_current(trial, annotation_filename, total_frm, timeline_plot, parsed_behavs):
    cv2.imwrite(f"{trial}_annotation_result.jpg", total_frm)
    cv2.imwrite(f"{trial}_behaviors_visualized.jpg", timeline_plot)

    annot_of = open(annotation_filename, 'w')
    print("Behavior Type\tStart Frame\tEnd Frame", file=annot_of)

    # write all the behavior epochs at the time to the file
    for behav, start, end in parsed_behavs:
        print(f"{behav}\t{start}\t{end}", file=annot_of)
    annot_of.close()

def add_warning(frm, message):
    # get charcter length of warning message and break it to fit the screen
    # while respecting spacings
    split = message.split(' ')
    word_idx = 0
    char_loc = 0
    prt_times = 0

    while char_loc < len(message):
        prtstr = ""
        char_count = 0
        while True:
            if word_idx == len(split):
                break

            word = split[word_idx]
            word_len = len(word)
            if (char_count + word_len) <= 20:
                if word_idx == 0:
                    char_loc += word_len
                else:
                    char_loc += (word_len + 1)

                if prtstr == "":  
                    prtstr += word
                    char_count += word_len

                else:
                    prtstr += f" {word}"
                    char_count += (word_len + 1)

                word_idx += 1

            else:
                break

        cv2.putText(frm, f"{prtstr}", (20,65+prt_times*30), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        prt_times += 1
        
    return frm
    
video_files = sorted(glob(os.path.join(DATA_DIR, "*.avi")))
trials = [os.path.basename(video_file).split(".avi")[0] for video_file in video_files]
print(f"[INFO] Found {len(trials)} trials: {trials}")

# start looping
for i, video_file in enumerate(video_files):
    trial = trials[i]
    print(f"\n[INFO] Start annotating trial: '{trial}'")
    print(f"[INFO] Setting ROI for video: {video_file}")
    ROI_FILE = os.path.join(".", f"{trial}_ROI.pkl")
    FRAMES_DIR = os.path.join(".", "frames", trial)

    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
    
    if not os.path.exists(ROI_FILE):
        vid = cv2.VideoCapture(video_file)
        _, first_frame = vid.read()
        vid.release()
        roi = set_roi(first_frame)
        joblib.dump(roi, ROI_FILE)
    else:
        print("[INFO] Using previously saved ROI info")
        roi = joblib.load(ROI_FILE)

    ## Generate frames first: to accelerate annotation control
    cap = cv2.VideoCapture(video_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if frames already exist
    frames = sorted(glob(os.path.join(FRAMES_DIR, "frame*.jpg")))
    if len(frames) == 0:
        last_frame = 0
    else:
        last_frame = int(re.match(r'frame([0-9]+)*.jpg', os.path.basename(frames[-1])).group(1))

    if last_frame == n_frames and not OVERWRITE:
        print(f"[INFO] Frames for trial '{trial}' are already generated")
    else:
        print(f"[INFO] Frames for trial '{trial}' will be generated first")
        print(f"[INFO] First frame to generate is frame #{last_frame+1}")

        # Generate frames in advance
        frame_n = last_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)

        while frame_n <= n_frames:
            if frame_n % 500 == 0:
                print(f"  [PROGRESS] Frame: {frame_n}/{n_frames}")

            ret, frm = cap.read()
            if not ret:
                break

            cropped = crop_roi(frm, roi)
            adjusted = adjust_gamma(cropped, gamma=1)
            adjusted = change_brightness(adjusted, value=10)
            resized = cv2.resize(adjusted, (int(adjusted.shape[1]*(VSPAN/adjusted.shape[0])), VSPAN), interpolation=cv2.INTER_LINEAR)
            cv2.putText(resized, f"FRAME {frame_n}", (20,30), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imwrite(os.path.join(FRAMES_DIR, f"frame{frame_n:05}.jpg"), resized)
            frame_n += 1

        cap.release()

    # initialization
    pos = 0
    frm = cv2.imread(os.path.join(FRAMES_DIR, f"frame{(pos+1):05}.jpg"))
    frame_rate = 20
    direction = 'f'
    status = 'stay'
    mode = 'player' # mode: player/annotator
    behav_selected = False
    warning = False
    warning_message = ""
    warning_count = 0
    
    ff = cv2.imread(os.path.join(FRAMES_DIR, f"frame{1:05}.jpg"))
    behavior_annotations = []
    parsed_behavs = []

    ANNOTATION_OUTFILE = f"{trial}_annotations.txt"
    ANNOTATIONS_LOADED = False
    # if annotations for the trial already exist, initialize using them
    if Path(ANNOTATION_OUTFILE).is_file():
        print(f"[INFO] Annotations for trial '{trial}' already exists")
        print("[INFO] The pre-existing annotations will be loaded")
        ANNOTATIONS_LOADED = True
        
        # skip the first line & split
        with open(ANNOTATION_OUTFILE, 'r') as f:
            parsed_behavs = [line[:-1].split('\t') for line in f.readlines()[1:]]
        
        # convert start/end frame numbers into integers
        parsed_behavs = [[behav, int(start), int(end)] for behav, start, end in parsed_behavs]
        
        # make behavior_annotations using parsed behaviors (reverse conversion)
        behavior_annotations = []
        for behav, start, end in parsed_behavs:
            behavior_annotations.append([start, f"{behav}/S"])
            behavior_annotations.append([end, f"{behav}/E"])
    
    timeline_plot = get_timeline_img(parsed_behavs, behaviors, ff.shape[1] + ANNOT_WINDOW_WIDTH)
    
    
    # make the common part of annotation in advance
    annot_common = np.zeros((ff.shape[0], ANNOT_WINDOW_WIDTH, 3), np.uint8)
    cv2.putText(annot_common, "-ANNOTATIONS-", (20,30+INTERVAL_L*2), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, (annotation, description) in enumerate(annotations):
        cv2.putText(annot_common, f"{annotation}{' '*(5-len(annotation))}{description}", (20,INTERVAL_S+INTERVAL_L*2+INTERVAL_S*(i+1)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    end_pos = INTERVAL_S+INTERVAL_L*2+INTERVAL_S*len(annotations)

    cv2.putText(annot_common, "-COMMANDS-", (20,end_pos+INTERVAL_L), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for i, (key, command) in enumerate(commands):
        cv2.putText(annot_common, f"{key}{' '*(7-len(key))}{command}", (20,end_pos+INTERVAL_L+INTERVAL_S*(i+1)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    end_pos = end_pos+INTERVAL_L+INTERVAL_S*len(commands) 
    
    pos_bkup = 0
    behavior = ''
    behav_to_save = ''
    
    # loop until break
    while True:
        if pos == n_frames:
            pos = 0

        # Read in a frame at current index, only if it has changed
        if pos != pos_bkup:
            frm = cv2.imread(os.path.join(FRAMES_DIR, f"frame{(pos+1):05}.jpg"))
        
        # update backup position
        pos_bkup = pos
        
        # Make an annotation area and annotate it
        annot = annot_common.copy()
        if direction == 'f':
            cv2.putText(annot, "Forward", (20,30), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif direction == 'b':
            cv2.putText(annot, "Backward", (20,30), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annot, f"Frame Rate: {frame_rate}", (20,30+INTERVAL_L), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(annot, f"MODE: {mode}", (20,end_pos+INTERVAL_L), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        current_behav = "None"
        for behav, start, end in parsed_behavs:
            if start <= pos+1 <= end:
                current_behav = behav
                break
        
        cv2.putText(annot, f"Annotated as: {current_behav}", (20,end_pos+2*INTERVAL_L), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if mode == 'annotator':
            if behav_selected:
                c = (255,255,255)
            else:
                c = (0,0,255)
            cv2.putText(annot, f"behavior: {behavior}", (20,end_pos+3*INTERVAL_L), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)
        
        if mode == 'saver':
            if behav_selected:
                c = (255,255,255)
            else:
                c = (0,0,255)
            cv2.putText(annot, f"behavior: {behav_to_save}", (20,end_pos+3*INTERVAL_L), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)

        # Horizontal concat the two
        concat_frm = cv2.hconcat([frm, annot])

        # Vertical concat between horizontal concat & timeline plot
        total_frm = cv2.vconcat([concat_frm, timeline_plot])
        
        if (pos == 0) and ANNOTATIONS_LOADED:    
            save_current(trial, ANNOTATION_OUTFILE, total_frm, timeline_plot, parsed_behavs)
            # to make save_current() not run repeatedly
            ANNOTATIONS_LOADED = False
        
        # show warning message on the annotator for five iterations
        if warning :
            total_frm = add_warning(total_frm, warning_message)          
            warning_count += 1
            if warning_count >= 5:
                warning = False
                warning_message = ""
                warning_count = 0

        resize_values = (total_frm.shape[1]//REDUCE_SIZE_RATIO, total_frm.shape[0]//REDUCE_SIZE_RATIO)
        total_frm_resized = cv2.resize(total_frm, resize_values) 
        cv2.imshow('Behavior Annotation', total_frm_resized)

        # Adding this might help smoothly exiting loop
        cv2.startWindowThread()

        key = cv2.waitKey(int((1/frame_rate)*1000))

        if mode == 'annotator':
            if key == -1:
                continue

            # when enter is pressed again
            if key == 13:
                mode = 'player'
                behav_selected = False
                print("[INFO] Exiting the annotator mode...")
                continue

            key_chr = chr(key)

            # check if the pressed key is an alphabet
            if not key_chr.isalpha():
                print("[WARNING] Pressed key is not an alphabet")
                
                warning = True
                warning_message = "[WARNING] Pressed key is not an alphabet"
                continue

            if key_chr.islower():
                key_chr = key_chr.upper()
            
            # handle 'z/Z' for deleting one letter in the behavior code
            if key_chr == 'Z':
                if behavior.endswith('/'):
                    behavior = ''
                    behav_selected = False
                else:
                    behavior = behavior[:-1]
                continue
            
            if behav_selected:
                if key_chr not in ['S', 'E']:
                    print("[WARNING] After selecting a behavior, press either 's/S' or 'e/E' only")
                    print("[INFO] To exit the annotator mode, press enter")
                    
                    warning = True
                    warning_message = "[WARNING] After selecting a behavior, press either 's/S' or 'e/E' only"
                else:
                    annotation_str = behavior + key_chr
                    
                    # when the position to add is already occupied by a pre-registered behavior
                    if any(start <= pos+1 <= end for start,end in \
                           [[start, end] for behav, start, end in parsed_behavs]):
                        print(f"[INFO] An annotation that contains this position ({pos+1}) already exists")
                        print("[INFO] The mentioned annotation will be deleted to add the newly requested annotation")
                        behavior_annotations, parsed_behavs = remove_behav_epoch(behavior_annotations, parsed_behavs, pos+1)
                        timeline_plot = get_timeline_img(parsed_behavs, behaviors, frm.shape[1] + ANNOT_WINDOW_WIDTH)
                          
                    behavior_annotations.append([pos+1, annotation_str])
                    print(f"[INFO] Behavior annotation '{annotation_str}' added for frame {pos+1}")
                    
                    # update timeline plot when a behavior has ended
                    if annotation_str.endswith('E'):
                        try:
                            parsed_behavs = parse_annotations(behavior_annotations, behaviors)
                        except AssertionError:
                            print("[ERROR] An error occurred during parsing the annotations")
                            print("[ERROR] A start of a behavior must be accompanied by an end of that behavior")
                            print("[ERROR] Latest annotation(s) will be removed")
                            
                            warning = True
                            warning_message = "[ERROR] An error occurred during parsing the annotations"
                            
                            behavior_annotations.pop()
                            
                            idx = len(behavior_annotations)-1
                            while True:
                                if idx < 0:
                                    # if no behavior is left
                                    break
                                if behavior_annotations[idx][1].endswith('S'):
                                    del(behavior_annotations[idx])
                                    idx -= 1
                                else:
                                    break
                            parsed_behavs = parse_annotations(behavior_annotations, behaviors)
                            
                            behavior = ''
                            behav_selected = False
                            timeline_plot = get_timeline_img(parsed_behavs, behaviors, frm.shape[1] + ANNOT_WINDOW_WIDTH)
                            continue
                        
                        timeline_plot = get_timeline_img(parsed_behavs, behaviors, frm.shape[1] + ANNOT_WINDOW_WIDTH)
                    
                    # at this line, either 'S' or 'E' was selected -> changing to the player mode
                    behav_selected = False
                    mode = 'player'
                    print("[INFO] Exiting the annotator mode...")

            else:
                # handle 'd/D' for removing the behavior code(s) that exist at the current pointer
                if key_chr == 'D':
                    print("[INFO] 'D' key was pressed. Behavior epoch existing at this position will be deleted")
                    behavior_annotations, parsed_behavs = remove_behav_epoch(behavior_annotations, parsed_behavs, pos+1)
                    timeline_plot = get_timeline_img(parsed_behavs, behaviors, frm.shape[1] + ANNOT_WINDOW_WIDTH)
                    mode = 'player'
                    print("[INFO] Exiting the annotator mode...")
                    continue

                behav_candidate = behavior + key_chr

                if not any(annotation.startswith(behav_candidate)\
                           for annotation in behaviors):
                    print(f"[ERROR] Behavior code '{behav_candidate}' doesn't match with provided annotations")
                    
                    warning = True
                    warning_message = f"[ERROR] Behavior code '{behav_candidate}' doesn't match with provided annotations"
                    continue

                else:
                    behavior += key_chr

                if behavior in behaviors:
                    print(f"[INFO] Selected behavior: {behavior}")
                    behavior += '/'
                    behav_selected = True


        elif mode == 'player':
            try:
                if key == -1:
                    new_status = status
                else:
                    new_status = commands_dict[key]

                if new_status == 'play/stop':
                    if status == 'stay':
                        status = 'play'
                    elif status == 'play':
                        status = 'stay'
                elif new_status == 'play':
                    if direction == 'f':
                        pos += 1
                    elif direction == 'b':
                        if pos != 0:
                            pos -= 1
                        else:
                            status = 'stay'
                elif new_status == 'forward':
                    direction = 'f'
                elif new_status == 'backward':
                    direction = 'b'
                elif new_status == 'prev_frame':
                    if pos != 0:
                        pos -= 1
                    else:
                        pos = n_frames-1
                    status = 'stay'
                elif new_status == 'next_frame':
                    pos += 1
                    status = 'stay'
                elif new_status == 'jump_frame':
                    pos += FRAME_JUMP_SIZE
                    if pos > n_frames:
                        pos -= n_frames
                elif new_status == 'slower':
                    frame_rate = max(frame_rate-FR_INTERVAL, 5)
                elif new_status == 'faster':
                    frame_rate = min(MAX_FR, frame_rate+FR_INTERVAL)
                elif new_status == 'save':
                    print(f"[INFO] Saving current state of annotations for trial {trial}")
                    save_current(trial, ANNOTATION_OUTFILE, total_frm, timeline_plot, parsed_behavs)
                elif new_status == 'video':
                    mode = 'saver'
                    behav_to_save = ''
                    epochs_to_save = []
                    print("[INFO] Entering the saver mode...")
                elif new_status == 'annotate':
                    mode = 'annotator'
                    behavior = ''
                    print("[INFO] Entering the annotator mode...")
                elif new_status == 'exit':
                    cv2.destroyWindow('Behavior Annotation')
                    # An additional cv2.waitkey() call is needed
                    # after destroy call to correctly destroy window(s)
                    cv2.waitKey(1)
                    break

            except KeyError:
                print(f"[WARNING] Invalid Key was pressed: {chr(key)}")
                
                warning = True
                warning_message = f"[WARNING] Invalid Key was pressed: {chr(key)}"
                
        elif mode == 'saver':
            # when behavior to save is determined, 
            # halt the player and open a new window showing frames being saved
            # when all the savings are done, continue the loop
            
            if behav_selected:
                # for each epoch
                for behav, epoch_start, epoch_end in epochs_to_save:
                    # loop from start to end, save a video
                    
                    # preparation for video synthesis
                    fps = VIDEO_SAVE_FPS
                    h, w, _ = total_frm.shape
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    out_file = os.path.join(VIDEO_SAVE_DIR, f"{behav}_{trial}_synthesized_video_[{epoch_start}-{epoch_end}].avi")

                    out = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
                    
                    for save_fn in range(epoch_start, epoch_end+1):
                        # 1. video_frame
                        save_frm = cv2.imread(os.path.join(FRAMES_DIR, f"frame{(save_fn):05}.jpg"))
                        
                        # annotate current behavior
                        current_behav = ""
                        for behav, start, end in parsed_behavs:
                            if start <= save_fn <= end:
                                current_behav = behav
                                break
                        
                        cv2.putText(save_frm, f"Current Behavior: {current_behav}", (20,65), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # 2. annotation area
                        annot = np.zeros((save_frm.shape[0], ANNOT_WINDOW_WIDTH, 3), np.uint8)
                        cv2.putText(annot, "-ANNOTATIONS-", (20,INTERVAL_L_SAVER), \
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        for i, (annotation, description) in enumerate(annotations):
                            cv2.putText(annot, f"{annotation}{' '*(5-len(annotation))}{description}", (20,INTERVAL_L_SAVER+INTERVAL_S_SAVER*(i+1)), \
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        end_at = INTERVAL_L_SAVER+INTERVAL_S_SAVER*len(annotations)+50
                        
                        cv2.putText(annot, "-INFORMATION-", (20, end_at + INTERVAL_L_SAVER), \
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        if behav_to_save == 'T':
                            cv2.putText(annot, f"Behavior: All", (20, end_at + 2*INTERVAL_L_SAVER), \
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            cv2.putText(annot, f"Behavior: {behav}", (20, end_at + 2*INTERVAL_L_SAVER), \
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(annot, f"Epoch: [{epoch_start}-{epoch_end}]", (20, end_at + 3*INTERVAL_L_SAVER), \
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # 3. make total frame to save (using frm, annot, timeline_plot)
                        
                        # Horizontal concat between frm & annot
                        concat_frm = cv2.hconcat([save_frm, annot])

                        # Vertical concat between horizontal concat & timeline plot
                        save_frm = cv2.vconcat([concat_frm, timeline_plot])
                        
                        out.write(save_frm)
                        cv2.imshow('Saving Frames', save_frm)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if Path(out_file).is_file():
                        print(f"[INFO] Successfully saved video: {os.path.basename(out_file)}")
                    out.release()
                    
                # when all the videos are generated
                cv2.destroyWindow('Saving Frames')
                # An additional cv2.waitkey() call is needed
                # after destroy call to correctly destroy window(s)
                cv2.waitKey(1)
                
                behav_selected = False
                behav_to_save = ''
                mode = 'player'
                print("[INFO] Exiting the saver mode...")
                continue
                
            else:
                # when no key was pressed
                if key == -1:
                    continue

                key_chr = chr(key)

                # check if the pressed key is an alphabet
                if not key_chr.isalpha():
                    print("[WARNING] Pressed key is not an alphabet")

                    warning = True
                    warning_message = "[WARNING] Pressed key is not an alphabet"
                    continue

                if key_chr.islower():
                    key_chr = key_chr.upper()

                # handle 'z/Z' for deleting one letter in the behavior code
                if key_chr == 'Z':
                    behav_to_save = behav_to_save[:-1]
                    continue
                
                # when saving the total video
                elif key_chr == 'T':
                    behav_to_save = 'T'
                    print("[INFO] Selected behavior to save: Total")
                    epochs_to_save = [['T', 1, n_frames]]
                    behav_selected = True
                    continue
                    
                elif key_chr == 'A':
                    behav_to_save = 'A'
                    print("[INFO] Selected behavior to save: All")
                    epochs_to_save = parsed_behavs
                    behav_selected = True
                    continue
                    
                behav_candidate = behav_to_save + key_chr

                if not any(annotation.startswith(behav_candidate)\
                           for annotation in behaviors):
                    print(f"[ERROR] Behavior code '{behav_candidate}' doesn't match with provided annotations")
                    
                    warning = True
                    warning_message = f"[ERROR] Behavior code '{behav_candidate}' doesn't match with provided annotations"
                    continue

                else:
                    behav_to_save += key_chr

                if behav_to_save in behaviors:
                    print(f"[INFO] Selected behavior to save: {behav_to_save}")
                    epochs_to_save = [[behav, start, end] for behav, start, end in parsed_behavs \
                                      if behav == behav_to_save]
                    behav_selected = True
            
    save_current(trial, ANNOTATION_OUTFILE, total_frm, timeline_plot, parsed_behavs)
    