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
    import scipy.ndimage as ndimage
    
    T, M, N = masks.shape[:3]
    
    all_tra_contour = []
    
    for tra_ii, tra in enumerate(annot_track):
        tra_contour = []
        
        tra_id = tra[0]
        start = tra[1]
        end = tra[2]
        
        for tt in np.arange(start,end+1):
            obj_mask = masks[tt]==tra_id
            obj_mask = ndimage.gaussian_filter(obj_mask*255, sigma=5)
            obj_mask = obj_mask / obj_mask.max()
            
            contour = skmeasure.find_contours(obj_mask, level=0.5)
            
            try:
                contour = contour[np.argmax([len(cc) for cc in contour])] #  retain the largest - just in case. 
                if n_boundary_pts is not None:
                    contour = resample_curve(contour[:,0], contour[:,1], k=1, s=None, n_samples=n_boundary_pts)
                    contour[:,0] = np.clip(contour[:,0], 0, M-1)
                    contour[:,1] = np.clip(contour[:,1], 0, N-1)
                    contour = contour[:,::-1] # for (x,y) convention
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
            coords = coords[:,::-1] # flip to (x,y) convention
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
        pickle.dump(a, handle)
    return []
    
def load_pickle(savefile):
    import pickle
    with open(savefile, 'rb') as handle:
        b = pickle.load(handle)
        
    return b


def parse_division_events(annot_track):
    
    import numpy as np
    
    all_div_xy = []
    
    for tra in annot_track:
        tra_xy = []
        
        tra_id = tra[0]
        start = tra[1]
        end = tra[2]
        lineage=tra[3]
        
        if lineage > 0:
            # then it has a legit mother. 
            mum_dat = annot_track[annot_track[:,0]==lineage][0]
            # print(mum_dat)
            # print(mum_dat[2], start, lineage)
            if start - mum_dat[2] == 1:
                all_div_xy.append(start)
            else:
                all_div_xy.append(-1)
        else:
            all_div_xy.append(-1)
                
    all_div_xy = np.hstack(all_div_xy)
        
    return all_div_xy

def identify_border_tracks(contour_xy_array, shape, percent=0.1, border_pad=10):
    
    import numpy as np 
    
    border_x = np.logical_or(contour_xy_array[...,0] <= border_pad, 
                             contour_xy_array[...,0] >= shape[1]-border_pad)
    border_y = np.logical_or(contour_xy_array[...,1] <= border_pad, 
                             contour_xy_array[...,1] >= shape[0]-border_pad)
    
    nan_mask = np.isnan(contour_xy_array[...,0]) # 
    border_contour = np.logical_or(border_x, border_y) # only needs one 
    border_contour[nan_mask] = 0
    
    # ok now we aggregate.
    border_contour = border_contour.astype(np.float32)
    border_contour = np.sum(border_contour, axis=-1) / float(border_contour.shape[-1])  # summing over the number of boundaries.
    border_contour = border_contour > percent
    
    return border_contour
    
    
if __name__=="__main__":
    
    
    """
    This script demonstrates how to prepare a video dataset from the single-cell tracking challenge which comes with reference segmentations and tracking for SPOT analysis using our library.
    
    These scripts were used to generate the example data. 
    
    For this example we use the U373 dataset which can be downloaded from the following URL:
        http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip
    
    """
    
    import numpy as np 
    import glob 
    import os 
    import pylab as plt 
    
    import SPOT.Utility_Functions.file_io as fio 
    
    
    n_boundary_pts = 200 # same as the paper. This is the number of boundary points we model the cell shape with
    
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
    
    
    for cellfolder in cellfolders[:]: 
        
        
        infolder = os.path.join(rootfolder,cellfolder)
        rootname, basename = os.path.split(cellfolder)
        
        """
        The scripts to parse the dataset. - Test this on several cell tracking challenge dataset 
        """
        results = parse_dataset(infolder)
        raw = read_img_sequence(results['raw_files'])
        tra_xy = get_xy_coords(results['annot_track'], 
                                results['seg_masks'])
        # tra_xy = get_xy_coords(results['annot_track'], 
        #                         results['annot_masks'])
        
        plt.figure()
        plt.imshow(raw[0])
        plt.show()
        
        contour_xy = get_contours(results['annot_track'], 
                                    results['seg_masks'],  # 'seg_masks'
                                    n_boundary_pts=n_boundary_pts)
        
        """
        We extract the contours in (x,y) coordinates 
        """
        contour_xy_array = pad_tracks( contour_xy, 
                                        len(raw), 
                                        boundary_samples=n_boundary_pts)
    
        
        """
        parse out division -> as an extra metadata!. 
        """
        divisions = parse_division_events(results['annot_track'])
        
        print(divisions)
        
        """
        mark border boundaries
        """
        """
        In addition mark all the contours on the boundary of the image and should be removed for analysis as it creates artificial eccentricity! 
        """
        
        contour_xy_array_border = identify_border_tracks(contour_xy_array, 
                                                         shape=raw.shape[1:3])
        contour_xy_array_filter = contour_xy_array.copy()
        contour_xy_array_filter[contour_xy_array_border>0] = np.nan
        
        # now we can save this. 
        results['raw'] = raw
        results['tra_xy'] = tra_xy
        results['contour_xy_array'] = contour_xy_array
        results['contour_xy_array_filter'] = contour_xy_array_filter
        results['div_times'] = divisions
        
        
        saveresultsfolder = os.path.join(saverootfolder, rootname, basename); fio.mkdir(saveresultsfolder)
        saveresultsfile = os.path.join(saveresultsfolder, 'seg_boundaries_and_tracks.pickle')
        
        # write this. 
        write_pickle(saveresultsfile, results)
        
       
        """
        Add the additional parsed into the dictionary and save out for easy use. 
        """
        
        saveresultsfolder_plot = os.path.join(saverootfolder, rootname, basename+'_SAM', 'raw_seg_masks_overlay'); 
        fio.mkdir(saveresultsfolder_plot)
        
        
        import seaborn as sns 
        color_tracks = sns.color_palette('Spectral',
                                         n_colors=len(contour_xy_array_filter))
        
        
        # checking with a plot. 
        for tt in np.arange(len(raw)):
            fig, ax = plt.subplots(figsize=(10,10))

            plt.imshow(raw[tt], cmap='gray')

            # for tra in tra_xy:
            #     plt.plot(tra[:,2], 
            #               tra[:,1], lw=3)
                
            # for cont in contour_xy[:]:
            #     for cc in cont[:]:
            #         tt_tp, cc_tt = cc 
            #         if tt_tp == tt:
            #             plt.plot(cc_tt[:,1], 
            #                       cc_tt[:,0],'.', lw=3)
                        
            # for cont in contour_xy_array_filter[:,tt]:
            for cont_ii in np.arange(len(contour_xy_array_filter[:,tt])):
                cont = contour_xy_array_filter[cont_ii,tt]
            # for cont in contour_xy_array[:,tt]:
                # for cc in cont[:]:
                if np.isnan(cont[0,0]):
                    pass
                else:
                # tt_tp, cc_tt = cc 
                # if tt_tp == tt:
                    plt.plot(cont[:,0], 
                             cont[:,1],'.', lw=3, color=color_tracks[cont_ii])

            plt.axis('off')
            plt.grid('off')
            plt.savefig(os.path.join(saveresultsfolder_plot, 
                                     str(tt).zfill(3)+'.png'), dpi=120, bbox_inches='tight')
            plt.show()
            
    
    
    
    