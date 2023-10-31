# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:37:14 2023

@author: fyz11
"""

def read_img_sequence(files):
    
    import skimage.io as skio 
    
    return np.array([skio.imread(ff) for ff in files])

def read_txt(fpath):
    """ 
    read a text file
    """
    import numpy as np
    array = []
    with open(fpath,'r') as f:
        for line in f:
            array.append(np.hstack(line.strip('\n').split()).astype(np.int32))
            
    return np.vstack(array) 

def parse_dataset(fpath):
    """
    Load and return the segmentation masks and tracks 

    Returns
    -------
    dict: 
        
    """
    import glob 
    import os 
    import numpy as np 
    import skimage.io as skio 
    import skimage.measure as skmeasure 
    
    rawfiles = np.sort(glob.glob(os.path.join(fpath, '*.tif')))
    
    # load and read in the manual segmentation masks - not always present.... 
    try:
        maskfiles = np.sort(glob.glob(os.path.join(fpath+'_ST', 'SEG', '*.tif')))
        masks = np.array([skio.imread(ff) for ff in maskfiles])
    except:
        maskfiles = []
        masks = []

    # print(masks.shape)
    # load the tracking 
    man_tra_file = os.path.join(fpath+'_GT', 'TRA', 'man_track.txt')
    man_tra_annot = read_txt(man_tra_file)
    
    # print(man_tra_annot)
    man_tra_mask_files = np.sort(glob.glob(os.path.join(fpath+'_GT', 'TRA', '*.tif')))
    man_tra_mask = np.array([skio.imread(ff) for ff in man_tra_mask_files])
    
    man_tra_seg_files = np.sort(glob.glob(os.path.join(fpath+'_GT', 'SEG', '*.tif')))
    man_tra_seg = np.array([skio.imread(ff) for ff in man_tra_mask_files])
    
    # print(man_tra_mask.shape)
    
    track_dict = {'raw_files': rawfiles,
                  'seg_files': maskfiles, 
                  'seg_masks': masks, 
                  'annot_maskfiles': man_tra_mask_files,
                  'annot_masks': man_tra_mask, 
                  'annot_segfiles': man_tra_seg_files,
                  'annot_segs': man_tra_seg, 
                  'annot_file': man_tra_file, 
                  'annot_track': man_tra_annot}
    
    return track_dict

def resample_curve(x,y, k=1, s=0, n_samples=10, per=True):
    
    import scipy.interpolate
    import numpy as np 

    if s is None:
        tck, u = scipy.interpolate.splprep([x,y], k=k, per=per)
    else:
        tck, u = scipy.interpolate.splprep([x,y], k=k, s=s, per=per)
    unew = np.linspace(0, 1.00, n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def get_contours(annot_track, masks, n_boundary_pts=None):
    
    import numpy as np 
    import skimage.measure as skmeasure
    
    T, M, N = masks.shape[:3]
    
    all_tra_contour = []
    
    for tra_ii, tra in enumerate(annot_track):
        tra_contour = []
        
        tra_id = tra[0]
        start = tra[1]
        end = tra[2]
        
        for tt in np.arange(start,end+1):
            contour = skmeasure.find_contours(masks[tt]==tra_id)
            
            try:
                contour = contour[np.argmax([len(cc) for cc in contour])] #  retain the largest - just in case. 
                if n_boundary_pts is not None:
                    contour = resample_curve(contour[:,0], contour[:,1], k=1, s=None, n_samples=n_boundary_pts)
                    contour[:,0] = np.clip(contour[:,0], 0, M-1)
                    contour[:,1] = np.clip(contour[:,1], 0, N-1)
            except:
                contour = np.hstack([np.nan, np.nan])[None,:] # we are missing an annotation!. 
                
            tra_contour.append([tt, contour])
        all_tra_contour.append(tra_contour)
        
    return all_tra_contour 


def get_xy_coords(annot_track, masks):
    
    import numpy as np
    
    all_tra_xy = []
    
    for tra in annot_track:
        tra_xy = []
        
        tra_id = tra[0]
        start = tra[1]
        end = tra[2]
        
        for tt in np.arange(start,end+1):
            coords = np.argwhere(masks[tt]==tra_id)
            tra_xy.append(np.hstack([tt, np.nanmean(coords, axis=0)]))
            
        tra_xy = np.vstack(tra_xy)
        all_tra_xy.append(tra_xy)
        
    return all_tra_xy 


def pad_tracks( contours, n_time, boundary_samples=100):
    
    import numpy as np 
    
    n_org = len(contours)
    boundary_ = np.zeros((n_org, n_time, boundary_samples, 2))
    boundary_[:] = np.nan
    
    for ii in range(n_org):
        # iterate over and fill in 
        cont_ii = contours[ii]
        
        for cc in cont_ii:
            tt, cc_tt = cc 
            
            if np.isnan(cc_tt[0,0]):
                pass
            else:
                boundary_[ii,tt] = cc_tt.copy()
                
    return boundary_


# these are required if writing much larger arrays or preserving numpy outputs. 
def write_pickle(savefile, a):
    import pickle
    with open(savefile, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL) # the protocol may cause probs for lower python versions.
    return []
    
def write_pickle_py3(savefile, a):
    import pickle
    with open(savefile, 'wb') as handle:
        pickle.dump(a, handle) # the protocol may cause probs for lower python versions.
    return []
    

def load_pickle(savefile):
    import pickle
    with open(savefile, 'rb') as handle:
        b = pickle.load(handle)
        
    return b


if __name__=="__main__":
    
    """
    This example shows how to use the parsed cell contour boundaries from the previous step to extract the shape, appearance and motion features separately per tracked cell.
    
    We will then compile the features in the next step.
    """
    
    
    import numpy as np 
    import glob 
    import os 
    import pylab as plt 
    from tqdm import tqdm 
    import skimage.exposure as skexposure 
    import scipy.io as spio 
    
    """
    loads for SPOT
    """
    import SPOT.Utility_Functions.file_io as fio 
    import SPOT.Tracking.optical_flow as SPOT_optical_flow
    import SPOT.Tracking.track as SPOT_track
    import SPOT.Features.features as SPOT_SAM_features
    

    n_boundary_pts = 200 # same as the paper. 
    # this is arbitrary unit i.e. we are just going to use the pixel 
    pixel_res = 1 # um 
    time_res = 1 # h
    
    
    # set an optional downsampling
    ds = 1. # to do: incorporate this into the saving. for future. 
    boundary_samples=200 # the target number of boundary samples, used originally in the tracking. 
    # n_files = len(masterimgfiles)
    # rev_channels_for_plot = True # whether to reverse color channels for plotting purposes
    
    # true... we should check and test thes. 
    border_pixel_size = 10
    border_pixel_frac = .1
    
    
    """
    Set up optical flow parameters, which is needed to compute motion-based features 
    """
    optical_flow_params = dict(pyr_scale=0.5, levels=5, winsize=5, iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
    
    
    """
    Specify the rootfolder where one has downloaded the single cell tracking challenge video dataset to 
    """
    rootfolder = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Paper\Suppl_GitHub_Code\test_cell-tracking-challenge_results\PhC-C2DH-U373_analysis'
    
    """
    Specify the top-level save folder we will save the intermediate outputs to.
    """
    saverootfolder = r'C:\Users\fyz11\Documents\Work\Projects\Lu-Organoids\Paper\Suppl_GitHub_Code\test_cell-tracking-challenge_results\PhC-C2DH-U373_analysis\SPOT'
    
    """
    Specify which of the videos will be analyzed. Here, both of the provided will be.
    """
    cellfolders = [r'PhC-C2DH-U373\01',  
                   r'PhC-C2DH-U373\02'] 
    
    
    for cellfolder in tqdm(cellfolders[:]): # this is the simulated. 
        
        print(cellfolder)
        
        infolder = os.path.join(rootfolder,cellfolder)
        rootname, basename = os.path.split(cellfolder)
        
        """
        Specify save folder location
        """
        saveresultsfolder = os.path.join(saverootfolder, rootname, basename); 
        saveresultsfile = os.path.join(saveresultsfolder, 'seg_boundaries_and_tracks.pickle')
        
        expt = rootname+'_'+basename
        
        """
        Grab the previous steps' results. 
        """
        results = load_pickle(saveresultsfile)
        
        # grab the contours and raw 
        raw = results['raw'].copy()
        contour_xy_array = results['contour_xy_array_filter'].copy()
        
        seg = results['seg_masks'].copy()
        #seg = results['annot_masks'].copy() # this is for the simulated. 
        
        """
        remove any boundaries that are too small. 
        """
        x_1 = np.min(contour_xy_array[...,0], axis=-1) # this needed to be flipped ... and changed. 
        x_2 = np.max(contour_xy_array[...,0], axis=-1)
        y_1 = np.min(contour_xy_array[...,1], axis=-1)
        y_2 = np.max(contour_xy_array[...,1], axis=-1)
        
        # check the sizes.
        w = x_2 - x_1
        h = y_2 - y_1
        
        # print(np.sum(w<1))
        # print(np.sum(h<1))
        
        """
        pre-extraction of optical flow.
        """
        img_ = np.array([skexposure.rescale_intensity(frame) for frame in raw])
        flow = SPOT_optical_flow.extract_vid_optflow(img_, 
                                                     optical_flow_params, 
                                                     rescale_intensity=True)
        
        """
        MOSES features and tables computation. 
        """
        
        # a) shape features
        print('computing shape metrics')
        metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_morphology_features(contour_xy_array[...,::-1], # takes (r,c) conventions 
                                                                                                      imshape=raw.shape[1:], 
                                                                                                      all_feats_compute=True, 
                                                                                                      contour_curve_feats=True, 
                                                                                                      curve_order=4, 
                                                                                                      curve_err=1., 
                                                                                                      geom_features=True,
                                                                                                      morph_features=True,
                                                                                                      fourier_features=True, 
                                                                                                      shape_context_features=True,
                                                                                                      norm_scm=True,
                                                                                                      n_ref_pts_scm=5,
                                                                                                      pixel_xy_res = pixel_res)
        
        
        savemetricfile = os.path.join(saveresultsfolder, expt+'_shape-feats.mat')
        spio.savemat(savemetricfile, {'expt': expt, 
                                      'analysisfile':saveresultsfile,
                                      'metric_names': metrics_labels, 
                                      'metric_norm_bool': metrics_norm_bool, 
                                      'metrics': metrics})
    
        # b) appearance features
        metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_texture_features(raw, 
                                                                                        contour_xy_array[...,::-1], 
                                                                                        use_gradient_vid=False,
                                                                                        compute_all_feats =True,
                                                                                        compute_intensity_feats=True,
                                                                                        compute_contour_intensity_feats=True,
                                                                                        n_contours=3,
                                                                                        n_angles=1,
                                                                                        angle_start=None,
                                                                                        compute_sift=True,
                                                                                        siftshape=(64,64),
                                                                                        compute_haralick=True,
                                                                                        haralick_distance=15)

        savemetricfile = os.path.join(saveresultsfolder, expt+'_appearance-feats.mat')
        spio.savemat(savemetricfile, {'expt': expt, 
                                      'analysisfile':saveresultsfile,
                                      'metric_names': metrics_labels, 
                                      'metric_norm_bool': metrics_norm_bool, 
                                      'metrics': metrics})
        
        # c) motion features
        metrics, metrics_labels, metrics_norm_bool = SPOT_SAM_features.compute_boundary_motion_features(flow, 
                                                                                        contour_xy_array[...,::-1], 
                                                                                        compute_all_feats =True,
                                                                                        compute_global_feats=True,
                                                                                        compute_contour_feats=True,
                                                                                        n_contour_feat_bins=8,
                                                                                        cnt_sigma=3., 
                                                                                        n_contours=3,
                                                                                        n_angles=1,
                                                                                        angle_start=None,
                                                                                        pixel_res=pixel_res,
                                                                                        time_res=time_res,
                                                                                        compute_sift_feats=True,
                                                                                        siftshape=(64,64))
        
        savemetricfile = os.path.join(saveresultsfolder, expt+'_motion-feats.mat')
        spio.savemat(savemetricfile, {'expt': expt, 
                                      'analysisfile':saveresultsfile,
                                      'metric_names': metrics_labels, 
                                      'metric_norm_bool': metrics_norm_bool, 
                                      'metrics': metrics})
        
        
        
        
    
    
    
    
    