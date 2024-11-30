############ INFO ############
# Author : Gyu-Hwan Lee (Korea Institute of Science and Technology, Seoul National University)
# Contact: gh.lee@kist.re.kr
# Written : 2024-11-12
# Description : Code for extracting ultrasonic vocalization traces from audio spectrogram
#               For your specific use, you might alter following parameters:
#                   n_maxima: number of high power frequency points to retain at each time (default: 7)
#                   max_sep_fidx: maximum separation in frequency that is allowed in connecting trajectory, in multiple of frequency step (default: 4)
#                   min_seg_len: minimum length trajectory pieces should pass to be retained as valid vocalization trajectory, in multiple of time step (default: 8)
#                   min_thickness: minimum thickness of stacked trajectory pieces for a time point to be recognized to have vocalization (default: 2)
##############################

import numpy as np

def find_maxima(spect, ts, n_maxima):
    maxima_tidxs = []
    maxima_fidxs = []

    for tidx, t in enumerate(ts):
        spect_slice = spect[:, tidx]
        fidxs = np.argsort(spect_slice)[-n_maxima:]

        maxima_tidxs += [tidx]*n_maxima
        maxima_fidxs += fidxs.tolist()

    return np.array(maxima_tidxs), np.array(maxima_fidxs)

def remove_isolates_iterative(spec_ts, spec_fs, max_tidxs, max_fidxs, fidx_thresh):
    continue_removal = True
    
    while continue_removal:
        
        nonisolate_max_tidxs = []
        nonisolate_max_fidxs = []
        
        for tidx in np.arange(1, len(spec_ts)-1):
            curr_fidxs = max_fidxs[max_tidxs == tidx].tolist()
            prev_fidxs = max_fidxs[max_tidxs == tidx-1].tolist()
            next_fidxs = max_fidxs[max_tidxs == tidx+1].tolist()

            for curr_fidx in curr_fidxs:
                min_fidx = curr_fidx - fidx_thresh
                max_fidx = curr_fidx + fidx_thresh
                if any(min_fidx <= fidx <= max_fidx for fidx in prev_fidxs+next_fidxs):
                    nonisolate_max_tidxs.append(tidx)
                    nonisolate_max_fidxs.append(curr_fidx)
                    
        if len(nonisolate_max_tidxs) == len(max_tidxs):
            continue_removal = False
        
        else:
            max_tidxs = np.array(nonisolate_max_tidxs)
            max_fidxs = np.array(nonisolate_max_fidxs)
            
    return max_tidxs, max_fidxs

def find_segments(tidxs, fidxs, fidx_thresh):
    segments = [] # detected segments
    selected_pidxs = [] # point indices included in any segment
    
    for pidx, (tidx, fidx) in enumerate(zip(tidxs, fidxs)):
        if pidx not in selected_pidxs:
            segment_tidxs = [tidx]
            segment_fidxs = [fidx]
            segment_pidxs = [pidx]
            
            continue_search = True
            search_tidx = tidx+1
            
            while continue_search:
                candidate_pidxs = np.where(tidxs == search_tidx)[0]
                
                # use only candidates not already included in other segments
                candidate_pidxs = candidate_pidxs[np.logical_not(np.isin(candidate_pidxs, selected_pidxs))]
                candidate_fidxs = fidxs[candidate_pidxs]
                
                min_fidx = segment_fidxs[-1] - fidx_thresh
                max_fidx = segment_fidxs[-1] + fidx_thresh
                
                candidate_fidxs = candidate_fidxs[np.logical_and(candidate_fidxs >= min_fidx,
                                                                 candidate_fidxs <= max_fidx)]
                
                if len(candidate_fidxs) == 0:
                    segments.append([segment_tidxs, segment_fidxs])
                    selected_pidxs += segment_pidxs
                    
                    continue_search = False
                else:
                    closest_candidate = np.argmin(np.abs(candidate_fidxs - segment_fidxs[-1]))
                    candidate_idx = np.where(fidxs[candidate_pidxs] == candidate_fidxs[closest_candidate])[0][0]
                    
                    segment_tidxs.append(tidxs[candidate_pidxs[candidate_idx]])
                    segment_fidxs.append(fidxs[candidate_pidxs[candidate_idx]])
                    segment_pidxs.append(candidate_pidxs[candidate_idx])
                    
                    search_tidx += 1
                    
    return segments

def find_segments_v2(tidxs, fidxs, fidx_thresh):
    # sort time and frequency: both in increasing order
    sort_order = np.lexsort((fidxs, tidxs))  # Sort by `a`, then by `b`
    tidxs_sorted = tidxs[sort_order]
    fidxs_sorted = fidxs[sort_order]
        
    segments = [] # detected segments
    selected_pidxs = [] # point indices included in any segment
    
    for pidx, (tidx, fidx) in enumerate(zip(tidxs_sorted, fidxs_sorted)):        
        if pidx not in selected_pidxs:
            segment_tidxs = [tidx]
            segment_fidxs = [fidx]
            segment_pidxs = [pidx]
            
            continue_search = True
            search_tidx = tidx+1
            
            while continue_search:
                candidate_pidxs = np.where(tidxs == search_tidx)[0]
                
                # use only candidates not already included in other segments
                candidate_pidxs = candidate_pidxs[np.logical_not(np.isin(candidate_pidxs, selected_pidxs))]
                candidate_fidxs = fidxs[candidate_pidxs]
                
                min_fidx = segment_fidxs[-1] - fidx_thresh
                max_fidx = segment_fidxs[-1] + fidx_thresh
                
                candidate_fidxs_select = candidate_fidxs[np.logical_and(candidate_fidxs >= min_fidx,
                                                                        candidate_fidxs <= max_fidx)]
                
                if len(candidate_fidxs_select) == 0:
                    segments.append([segment_tidxs, segment_fidxs])
                    selected_pidxs += segment_pidxs
                    
                    continue_search = False
                else:
                    bottom_most_candidate = np.argmin(candidate_fidxs_select)
                    candidate_idx = np.where(candidate_fidxs == candidate_fidxs_select[bottom_most_candidate])[0][0]
                    
                    segment_tidxs.append(tidxs[candidate_pidxs[candidate_idx]])
                    segment_fidxs.append(fidxs[candidate_pidxs[candidate_idx]])
                    segment_pidxs.append(candidate_pidxs[candidate_idx])
                    
                    search_tidx += 1
                    
    return segments

def merge_segments(segments, spect, length_thresh, fwidth_thresh):
    tidxs_collect = np.concatenate([tidxs for tidxs, _ in segments if len(tidxs) >= length_thresh])
    fidxs_collect = np.concatenate([fidxs for _, fidxs in segments if len(fidxs) >= length_thresh])
    
    tidxs_unique = np.unique(tidxs_collect)
    
    tidxs_merged = []
    fidxs_merged = []
    for tidx in tidxs_unique:
        select_fidxs = sorted(fidxs_collect[tidxs_collect == tidx])
        
        latest = -1
        for fidx in select_fidxs:
            if latest == -1:
                latest = fidx
                count = 1
                collect = [fidx]
            else:
                if fidx <= latest+2:
                    latest = fidx
                    count += 1
                    collect.append(fidx)
                else:
                    if count >= fwidth_thresh:
                        tidxs_merged.append(tidx)
                        amps = np.array([spect[i, tidx] for i in collect])
                        weight_avg = np.sum(amps/np.sum(amps) * np.array(collect))
                        fidxs_merged.append(np.round(weight_avg))
                    
                    latest = fidx
                    count = 1
                    collect = [fidx]
                    
        if count >= fwidth_thresh:
            tidxs_merged.append(tidx)
            amps = np.array([spect[i, tidx] for i in collect])
            weight_avg = np.sum(amps/np.sum(amps) * np.array(collect))
            fidxs_merged.append(np.round(weight_avg))
                    
    return np.array(tidxs_merged, dtype=int), np.array(fidxs_merged, dtype=int)
    
def extract_trajectory(spect, spect_fs, spect_ts, n_maxima=7, max_sep_fidx=4, min_seg_len=8, min_thickness=2, n_indent=0):
    # find strong intensity points at each timepoint
    print(f"{' '*n_indent}[Progress] Step 1: finding maxima points")
    maxima_tidxs, maxima_fidxs = find_maxima(spect, spect_ts, n_maxima)
    
    # remove isolates
    print(f"{' '*n_indent}[Progress] Step 2: removing isolates")
    nonisolate_tidxs, nonisolate_fidxs = remove_isolates_iterative(spect_ts, spect_fs, maxima_tidxs, maxima_fidxs, max_sep_fidx)

    # detect segments, remove short ones
    print(f"{' '*n_indent}[Progress] Step 3: finding segments")
    segments = find_segments_v2(nonisolate_tidxs, nonisolate_fidxs, max_sep_fidx)

    # merge neighboring segments
    print(f"{' '*n_indent}[Progress] Step 4: merging segments")
    distilled_tidxs, distilled_fidxs = merge_segments(segments, spect, min_seg_len, min_thickness)

    trajectory_ts = spect_ts[distilled_tidxs]
    trajectory_fs = spect_fs[distilled_fidxs]
    
    return trajectory_ts, trajectory_fs
