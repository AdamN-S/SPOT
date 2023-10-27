# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:00:02 2020

@author: felix
"""
import numpy as np 

def locate_experiments(infolder):
    
    import os 
    
    all_dirs = os.listdir(infolder)
    all_expts = [os.path.join(infolder,d) for d in all_dirs if os.path.isdir(os.path.join(infolder,d))]
    
    return all_expts
    

def dog_frame(im, sigma0=None, sigma1=3):
    
    from skimage.filters import gaussian
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    
    if sigma0 is None:
        frame0 = im.copy()
    else:
        frame0 = gaussian(im, sigma=sigma0, preserve_range=True)
        
    if sigma1 is None:
        frame1 = im.copy()
    else:
        frame1 = gaussian(im, sigma=sigma1, preserve_range=True)
    
        
    dog_frame = frame0 - frame1 ; dog_frame[dog_frame<0] = 0
    dog_frame = rescale_intensity(dog_frame)
    dog_frame = img_as_ubyte(dog_frame)
    
    return dog_frame
    
    
    
def multi_level_gaussian_thresh(vol, n_classes=3, n_samples=10000):
    
    # do we need to do stratified for performance?
    from sklearn.mixture import GaussianMixture
    from skimage.exposure import rescale_intensity
    
    model = GaussianMixture(n_components=n_classes)
    
    volshape = vol.shape[:2]
    
    if len(vol.shape) > 2:
        vals = rescale_intensity(vol).reshape(-1,vol.shape[-1]).astype(np.float)
    else:
        vals = rescale_intensity(vol).ravel()[:,None].astype(np.float)

    random_select = np.arange(len(vals))
    np.random.shuffle(random_select)
    
    if n_samples is None:
        X = vals.copy()
    else:
        X = vals[random_select[:n_samples]]
    model.fit(X)
    
    if len(vol.shape) > 2:
        labels_means = model.means_[:,-1].ravel()
    else:
        labels_means = model.means_.ravel()

    labels_order = np.argsort(labels_means)
    
    labels = model.predict(vals)
    labels_ = np.zeros_like(labels)
    
    for ii, lab in enumerate(labels_order):
        labels_[labels==lab] = ii
    
    labels_ = labels_.reshape(volshape)
    
    return labels_
    
def sobel_img(frame):
    
    from skimage.filters import sobel
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    
    sobel_frame = sobel(frame); 
    sobel_frame = img_as_ubyte(rescale_intensity(sobel_frame))
    
    return sobel_frame
    
def eccentricity_threshold_segs(binary, ecc_thresh=0.75):
    
    from skimage.measure import regionprops, label
    
    labelled = label(binary)
    uniq_ids = np.unique(labelled)[1:]
    props = regionprops(labelled)
    
    props_ecc = np.hstack([p.eccentricity for p in props])
    select = uniq_ids[props_ecc<=ecc_thresh]
    
    labelled_ = np.zeros_like(labelled)
    
    for idd in select:
        labelled_[labelled==idd] = idd
        
    return labelled_
    
def multi_level_kmeans_thresh(vol, n_classes=3, n_samples=10000):
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    model = KMeans(n_clusters=n_classes, init='k-means++')
    
    volshape = vol.shape[:2]
    
    if len(vol.shape) > 2:
        vals = vol.reshape(-1,vol.shape[-1])
    else:
        vals = vol.ravel()[:,None]
        
#    vals = StandardScaler().fit_transform(vals)
        
    random_select = np.arange(len(vals))
    np.random.shuffle(random_select)
    
    X = vals[random_select[:n_samples]]
    model.fit(X)
    
    if len(vol.shape) > 2:
        labels_means = model.cluster_centers_[:,-1].ravel()
    else:
        labels_means = model.cluster_centers_.ravel()
    labels_order = np.argsort(labels_means)
    
    labels = model.predict(vals)
    labels_ = np.zeros_like(labels)
    
    for ii, lab in enumerate(labels_order):
        labels_[labels==lab] = ii
    
    labels_ = labels_.reshape(volshape)
    
    return labels_

def grabcut_img(img, binary, niter=1, dilate=3):
    
    import cv2
    from skimage.measure import regionprops
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    from scipy.ndimage.morphology import binary_fill_holes
    
    bbox = regionprops(skmorph.binary_dilation(binary, skmorph.disk(dilate))*1)[0].bbox
    y1,x1,y2,x2 = bbox
    
    # attempt grabcut...
    rect = (int(x1), int(y1), int(x2-x1), int(y2-y1))
            
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    mask = np.zeros(img.shape)
    mask[int(y1):int(y2), int(x1):int(x2)] = 1
    mask = mask.astype(np.uint8)
    
    img = img_as_ubyte(rescale_intensity(img))
    
    output = cv2.grabCut(np.dstack([img, 
                                    img,
                                    img]),
                            mask, 
                            rect,
                            bgdModel,
                            fgdModel,
                            niter, 
                            0)
    
    seg = np.logical_or(output[0]==3, 
                        output[0]==1)
    seg = binary_fill_holes(seg)
    
    return seg, [x1,y1,x2,y2]
    
def grabcut_bbox(img, bbox, niter=1, dilate=3):
    
    import cv2
    from skimage.measure import regionprops, label
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    from scipy.ndimage.morphology import binary_fill_holes
    
    m,n =img.shape[:2]
    x1,y1,x2,y2 = bbox
    
    x1 = np.clip(x1 - dilate, 0, n-1)
    x2 = np.clip(x2 + dilate, 0, n-1)
    y1 = np.clip(y1 - dilate, 0, m-1)
    y2 = np.clip(y2 + dilate, 0, m+1)
    
    # attempt grabcut...
    rect = (int(x1), int(y1), int(x2-x1), int(y2-y1))
            
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    mask = np.zeros(img.shape)
    mask[int(y1):int(y2), int(x1):int(x2)] = 1
    mask = mask.astype(np.uint8)
    
#    img = img_as_ubyte(rescale_intensity(img))
    
    output = cv2.grabCut(img,
                        mask, 
                        rect,
                        bgdModel,
                        fgdModel,
                        niter, 
                        0)
    
    seg = np.logical_or(output[0]==3, 
                        output[0]==1)
    seg = binary_fill_holes(seg)
    
#    # return largest component.
#    labelled = label(seg)
#    uniq_reg = np.setdiff1d(np.unique(labelled), 0)
#    uniq_reg_areas = np.hstack([np.sum(labelled==re) for re in uniq_reg])
#    max_area_reg_id = uniq_reg[np.argmax(uniq_reg_areas)]
#    
#    seg = labelled == max_area_reg_id
    
    return seg, [x1,y1,x2,y2]

def grabcut_refine_regions(binary, img, dilate=3):
    
    from skimage.measure import label
    
    labelled = label(binary)
    uniq_regions = np.unique(labelled)[1:]
    refined_out = np.zeros_like(labelled)
    
    for re in uniq_regions:
        
        mask = labelled == re
        mask_new, mask_bbox_init = grabcut_img(img, mask, niter=1, dilate=dilate)
        
        if mask_new.sum() > 0:
            refined_out[mask_new] = re
            
    refined_out = skmorph.remove_small_objects(refined_out>0, 200)
    refined_out = label(refined_out)
            
    return refined_out    

def grabcut_refine_regions_bbox(bboxes, img, dilate=3, minsize_region=300, mode='overlap', overlap_box_thresh=0.05, erode_mask=True, erode_ksize=1, dilate_mask=3):
    
    from skimage.measure import label
    
    if mode== 'overlap':
        refined_out = np.zeros(img.shape[:2], dtype=np.int)
        
        counter = 1
        for bbox in bboxes:
            
    #        mask = labelled == re
            mask_new, mask_bbox_init = grabcut_bbox(img, bbox, niter=1, dilate=dilate)
            
            if mask_new.sum() > 0:
                
                # return largest component.
                labelled = label(mask_new)
                uniq_reg = np.setdiff1d(np.unique(labelled), 0)
                uniq_reg_areas = np.hstack([np.sum(labelled==re) for re in uniq_reg])
                max_area_reg_id = uniq_reg[np.argmax(uniq_reg_areas)]
                
                mask_new = labelled == max_area_reg_id
                
                refined_out[mask_new] = counter
                counter += 1
                
        refined_out = skmorph.remove_small_objects(refined_out>0, minsize_region)
        refined_out = label(refined_out)
        
    if mode== 'individual':
        bbox_first_frame_filt_GC = []
        seg_bbox_first_frame_GC = []
        
        for bbox in bboxes:
            
    #        mask = labelled == re
            mask_new, mask_bbox_init = grabcut_bbox(img, bbox, niter=1, dilate=dilate)
            
            if mask_new.sum() > 0:
                
                labelled = label(mask_new)
                uniq_reg = np.setdiff1d(np.unique(labelled), 0)
                uniq_reg_areas = np.hstack([np.sum(labelled==re) for re in uniq_reg])
                max_area_reg_id = uniq_reg[np.argmax(uniq_reg_areas)]
                
                mask_new = labelled == max_area_reg_id
                
                
                bbox_first_frame_filt_GC.append(mask_bbox_init)
                seg_bbox_first_frame_GC.append(mask_new>0)
        
        bbox_first_frame_filt_GC = np.vstack(bbox_first_frame_filt_GC)
        seg_bbox_first_frame_GC = np.array(seg_bbox_first_frame_GC)
        
        
        """
        Attempt to filter by cliques. 
        """
        # refine the segmentations -> they should not share overlap. !
        bbox_ious = bbox_iou_corner_xy(bbox_first_frame_filt_GC,bbox_first_frame_filt_GC)
        
        # find the cliques
        overlap_cliqs = []
        bbox_first_frame_filt_GC_filt = []
        seg_bbox_first_frame_GC_filt = []
        
        for jj in range(len(bbox_ious)):
            overlap_jj = bbox_ious[jj]
            overlap_ids = np.arange(len(overlap_jj))[overlap_jj >= overlap_box_thresh]
            
            other_ids = np.setdiff1d(overlap_ids, jj)
            if len(other_ids)==0:
                # pass. isolatory region.
                bbox_first_frame_filt_GC_filt.append(bbox_first_frame_filt_GC[jj])
                seg_bbox_first_frame_GC_filt.append(seg_bbox_first_frame_GC[jj])
            else:
                if len(overlap_cliqs) == 0:
                    overlap_cliqs.append(overlap_ids)
                else:
                    new_cliq = True
                    for cliq_ii, cliq in enumerate(overlap_cliqs):
                        intersect = np.intersect1d(cliq, overlap_ids)
                        
                        if len(intersect) > 0:
                            overlap_cliqs[cliq_ii] = np.unique(np.hstack([cliq, overlap_ids]))
                            new_cliq=False
                            break
                    if new_cliq == True:
                        overlap_cliqs.append(overlap_ids)
                            

        if len(overlap_cliqs)>0:
            import itertools
            
            # resolve the ambiguities.
            for cliq in overlap_cliqs[:]:
                cliq_boxes = bbox_first_frame_filt_GC[cliq]
                cliq_masks = seg_bbox_first_frame_GC[cliq]
                
#                        fig, ax = plt.subplots(nrows=1,ncols=len(cliq_masks), figsize=(15,7))
#                        for iii in range(len(cliq_masks)):
#                            ax[iii].imshow(cliq_masks[iii])
#                        plt.show()
                
                modified_ids = []
                # we now do all possible comparisons. 
                for iii, jjj in itertools.combinations(range(len(cliq_boxes)), 2):
#                            print(iii, jjj)
                    mask1 = cliq_masks[iii].copy()
                    mask2 = cliq_masks[jjj].copy()
                    
                    overlap1, overlap2, iou = iou_mask(mask1, 
                                                       mask2)
                    
                    if overlap1 > overlap2:
                        
                        mask1 = skmorph.binary_dilation(mask1, skmorph.disk(dilate_mask))
                        # then 1 is smaller and needs to be used to mask 2.
                        mask2 = np.logical_and(mask2, np.logical_not(mask1))
#                                mask2 = skmorph.binary_erosion(mask2, skmorph.disk(3))
                        modified_ids.append(jjj)
                        cliq_masks[jjj] = mask2.copy()
                        cliq_masks[iii] = mask1.copy()
                        
                    if overlap2 >= overlap1:
                        mask2 = skmorph.binary_dilation(mask2, skmorph.disk(dilate_mask))
                        # then 2 is smaller and needs to be used to mask 1.
                        mask1 = np.logical_and(mask1, np.logical_not(mask2))
#                                mask1 = skmorph.binary_erosion(mask1, skmorph.square(3))
                        modified_ids.append(iii)
                        cliq_masks[iii] = mask1.copy()
                        cliq_masks[jjj] = mask2.copy()
                        
                modified_ids = np.unique(modified_ids)
                
                if erode_mask == True:
                    for iii in modified_ids:
                        cliq_masks[iii] = skmorph.binary_erosion(cliq_masks[iii], 
                                                                  skmorph.disk(erode_ksize))         
                        
                # write the boxes and regions back.     
                bbox_first_frame_filt_GC_filt = np.vstack([bbox_first_frame_filt_GC_filt, cliq_boxes])
                seg_bbox_first_frame_GC_filt = np.vstack([seg_bbox_first_frame_GC_filt, cliq_masks])
 
        refined_out = np.zeros(img.shape[:2], dtype=np.int)
        
        counter = 1
        for jj in range(len(seg_bbox_first_frame_GC_filt)):
            mask = seg_bbox_first_frame_GC_filt[jj]
            if np.sum(mask) >= minsize_region:
                refined_out[mask>0] = counter
                counter += 1
        
            
    return refined_out
    

def assign_bbox_tracks_seg( tracks, segs, shape, largest_component=True, dilate_sigma=None):
    
    from skimage.draw import polygon
    from skimage.measure import find_contours
    import skimage.morphology as skmorph
    
    m, n = shape
#    binary = np.zeros(shape, dtype=np.bool)
    box_spixels = []
    
    for seg in segs:
        
        if largest_component == True:
            
            contour = find_contours(seg, level=0)
            contour = contour[ np.argmax([len(c) for c in contour])]
            
            seg_ = np.zeros_like(seg)
            rr,cc = polygon(contour[:,0], contour[:,1], shape=seg_.shape)
            seg_[rr,cc] = 1
            
            if dilate_sigma is not None:
                print('dilating')
                seg_ = skmorph.binary_dilation(seg_ > 0 , skmorph.disk(dilate_sigma))
            
            seg = seg_.copy()
        
        box_spixels.append(np.arange(len(tracks))[seg[tracks[:,0,0], tracks[:,0,1]]>0])
        
    return box_spixels

def entropy_seg_frame(frame_im, n_classes = 3, ksize=3):
    
    from skimage.filters.rank import entropy
    ent = entropy(frame_im, skmorph.square(ksize))
    ent_seg = multi_level_kmeans_thresh(ent, n_classes=n_classes, n_samples=10000)
    ent_seg = ent_seg == n_classes-1
    
    ent_seg = skmorph.binary_erosion(ent_seg, skmorph.square(2))
    ent_seg = skmorph.remove_small_objects(ent_seg, min_size=100)
#        seg0 = skmorph.binary_closing(seg0, skmorph.disk(3))
    ent_seg = binary_fill_holes(ent_seg)
    
    return ent_seg


"""
scripts to deduce boundaries.
"""
def delaunay_region_pts(pts, shape, outlier_thresh=2, edge_filter=None ):
    
    from scipy.spatial import Delaunay
    from skimage.draw import polygon, line_aa
    from skimage.draw import line_aa
    import networkx as nx
    from sklearn.metrics.pairwise import pairwise_distances
#>>> img = np.zeros((10, 10), dtype=np.uint8)
#>>> rr, cc, val = line_aa(1, 1, 8, 8)
#>>> img[rr, cc] = val * 255
    
    all_delaunay_lengths = []
    
    """ density filteration """
#    dist_pts = 
    

#    from sklearn.metrics.pairwise import pairwise_distances
    
    pts_dist = pairwise_distances(pts); 
    for ii in range(len(pts_dist)):
        pts_dist[ii,ii] = 1000.
        
    pts_dist = pts_dist <= 30.
#    pts_dist_sum = pts_dist.sum(axis=0) >=5
    A = nx.from_numpy_matrix(pts_dist)
#    
    comps = list(nx.connected_components(A))
    select = comps[np.argmax([len(c) for c in comps])]
    select = np.hstack(select)
#    pts_dist_sum = pts_dist.sum(axis=0) >=5
#    select = pts_dist_sum
    
#    print('helllloo', len(select))
#    print(outlier_length)
    mask = np.zeros(shape)
    
    if len(select) > 0:
        
        pts_dist_select = pts_dist[select].sum(axis=0) > 5
#        select = 
        pts = pts[np.hstack(select)][pts_dist_select]
        
#        print(len(pts))
        
        """ DELAUNAY """
        tri = Delaunay(pts) 
        
        for tri_pts in tri.simplices:
            
            tri_pts = pts[tri_pts]
            
            tri_lengths = np.hstack([np.linalg.norm(tri_pts[0] - tri_pts[1]), 
                                     np.linalg.norm(tri_pts[1] - tri_pts[2]),
                                     np.linalg.norm(tri_pts[2] - tri_pts[0])])
            all_delaunay_lengths.append(np.nanmedian(tri_lengths))
            
        all_delaunay_lengths = np.hstack(all_delaunay_lengths)
        
        mean_dist = np.nanmean(all_delaunay_lengths)
        std_dist = np.nanstd(all_delaunay_lengths)
        
        outlier_length = mean_dist + outlier_thresh* std_dist
    

        
        for tri_pts in tri.simplices:
            
            tri_pts = pts[tri_pts]
            
            tri_lengths = np.hstack([np.linalg.norm(tri_pts[0] - tri_pts[1]), 
                                     np.linalg.norm(tri_pts[1] - tri_pts[2]),
                                     np.linalg.norm(tri_pts[2] - tri_pts[0])])
                
#            tri_lengths_test = tri_lengths>= np.maximum(50., outlier_length)
            tri_lengths_test = tri_lengths >= outlier_length
            
            keep_triangle = np.sum(tri_lengths_test) == 0
    #        keep_triangle = 1
            
            # check kedges. 
            if edge_filter is not None:
    #            line1 = draw_line_mask(tri_pts[0], tri_pts[1], edge_filter.shape)
    #            line2 = draw_line_mask(tri_pts[1], tri_pts[2], edge_filter.shape)
    #            line3 = draw_line_mask(tri_pts[2], tri_pts[0], edge_filter.shape)
                
                line1 = edge_filter[np.linspace(tri_pts[0,0], tri_pts[1,0], 3*tri_lengths[0]).astype(np.int), 
                                    np.linspace(tri_pts[0,1], tri_pts[1,1], 3*tri_lengths[0]).astype(np.int)]
                line2 = edge_filter[np.linspace(tri_pts[1,0], tri_pts[2,0], 3*tri_lengths[1]).astype(np.int), 
                                    np.linspace(tri_pts[1,1], tri_pts[2,1], 3*tri_lengths[1]).astype(np.int)]
                line3 = edge_filter[np.linspace(tri_pts[2,0], tri_pts[0,0], 3*tri_lengths[2]).astype(np.int), 
                                    np.linspace(tri_pts[2,1], tri_pts[0,1], 3*tri_lengths[2]).astype(np.int)]
                
                line_keep = np.max(edge_filter[line1]) + np.max(edge_filter[line2]) + np.max(edge_filter[line3])
                line_keep = line_keep == 0
                keep_triangle = np.logical_and(keep_triangle, line_keep)
                
            if keep_triangle:
                # check the length distribution -> draw only if they are dont involve non extremal lengths. 
                rr, cc = polygon(tri_pts[:,0], tri_pts[:,1])
                mask[rr,cc] = 1
                
    else:
        print('empty select')
        
    return mask>0



#def resample_curve(x,y,s=0, n_samples=10):
#    
#    import scipy.interpolate
#    
#    tck, u = scipy.interpolate.splprep([x,y], s=s)
#    unew = np.linspace(0, 1.00, n_samples)
#    out = scipy.interpolate.splev(unew, tck) 
#    
#    return np.vstack(out).T

def resample_curve(x,y, k=1, s=0, n_samples=10, per=True):
    
    import scipy.interpolate
    if s is None:
        tck, u = scipy.interpolate.splprep([x,y], k=k, per=per)
    else:
        tck, u = scipy.interpolate.splprep([x,y], k=k, s=s, per=per)
    unew = np.linspace(0, 1.00, n_samples)
    out = scipy.interpolate.splev(unew, tck) 
    
    return np.vstack(out).T

def corner_cutting_smoothing(x,y, n_iters=1):
    
    poly = np.vstack([x,y]).T
    poly = poly[:-1]

    for ii in range(n_iters):
        
        m,n = poly.shape
        poly_ = np.zeros((2*m,n))
        
        poly_close = np.vstack([poly, poly[0]])
        poly_0 = poly_close[:-1].copy()
        poly_1 = poly_close[1:].copy()
        
        poly_[::2] = poly_0 * .75 + poly_1 * .25
        poly_[1::2] = poly_0 * .25 + poly_1 * .75
        
        poly = poly_.copy()
        
    return np.vstack([poly, poly[0]])

def iou_mask(mask1, mask2):
    
    intersection = np.sum(np.abs(mask1*mask2))
    union = np.sum(mask1) + np.sum(mask2) - intersection # this is the union area. 
    
    overlap = intersection / float(union + 1e-8)
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    
    return intersection / float(area1 + 1e-8), intersection / float(area2 + 1e-8), overlap 

def draw_poly_mask(pts,shape):
    
    from skimage.draw import polygon
    mask = np.zeros(shape, dtype=np.bool)
    rr,cc = polygon(pts[:,0], pts[:,1])
    mask[rr,cc] = 1
    
    return mask > 0
    
def iou_polygon_area(pts0, pts1,shape):
    
    poly1 = draw_poly_mask(pts0, shape)
    poly2 = draw_poly_mask(pts1, shape)
    
    return iou_mask(poly1, poly2)


def prop_core_tracks(tracks, track_ids0, dist_thresh):
    
    # do a pre-filter before propping?
    from sklearn.neighbors import NearestNeighbors
    
    nframes = tracks.shape[1]
    
    track_ids_time = [track_ids0]
    
    for frame in tqdm(range(nframes-1)[:]):
        
        all_tracks_pts = tracks[:,frame+1].copy()
        track_select_ids = np.arange(len(all_tracks_pts))
        
        last_ids = track_ids_time[-1] # always use the core defined tracks.
        
        other_ids = np.setdiff1d(track_select_ids, last_ids)
        
        if len(other_ids) > 0:
            # 
            model = NearestNeighbors(radius=dist_thresh)
            model.fit(all_tracks_pts[other_ids])
            
            neighbors_dist, neigh_ids = model.radius_neighbors(X=all_tracks_pts[last_ids], radius=None, return_distance=True)
            
            neigh_ids = np.hstack(neigh_ids)
            neighbors_dist = np.hstack(neighbors_dist)
            
#            print(neigh_ids)
            
            n_neighs = np.sum(neighbors_dist<dist_thresh)
            
            if n_neighs>0:
            
                valid_neigh_ids = neigh_ids[neighbors_dist<dist_thresh]
                valid_neigh_ids = np.unique(valid_neigh_ids)
                valid_neigh_ids = other_ids[valid_neigh_ids]
                
#            if len(valid_neigh_ids) > 0:                
                valid_neigh_ids = np.hstack([last_ids, valid_neigh_ids])
                track_ids_time.append(valid_neigh_ids)
            
        else:
            track_ids_time.append(last_ids)
#        
#        print(len(valid_neigh_ids))
#        print(len(track_ids_time))
#        
    return track_ids_time
        

# bbox segmentation filtering 
def filter_bboxes_segmentation( seg_mask, bboxes, thresh_prob=0.1):
    
    m, n = seg_mask.shape
    binary = seg_mask > 0 
    box_coverage = []
    
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        x1 = np.clip(x1, 0, n-1)
        y1 = np.clip(y1, 0, m-1)
        x2 = np.clip(x2, 0, n-1)
        y2 = np.clip(y2, 0, m-1)
        
        crop = binary[int(y1):int(y2), 
                      int(x1):int(x2)]
        box_coverage.append(np.nanmean(crop))
        
    box_coverage = np.hstack(box_coverage)
    
    return bboxes[box_coverage>=thresh_prob]


def remove_duplicates(mask_centroids, dist_thresh=5):
    
    from sklearn.metrics.pairwise import pairwise_distances
    import networkx as nx
    
    dist_matrix = pairwise_distances(mask_centroids) <= dist_thresh
    
    # convert to networkx. 
    G = nx.from_numpy_array(dist_matrix)
    components = list(nx.connected_components(G))
    
    core_centroids = []
    
    for c in components:
        c = np.hstack(c)
        core_centroids.append(np.mean(mask_centroids[c], axis=0))
        
    core_centroids = np.vstack(core_centroids)
    
    return core_centroids
    
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def nms_bbox(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        
#        print(ratio)
        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def bbox_iou_corner_xy(bboxes1, bboxes2):
    """
    computes the distance matrix between two sets of bounding boxes.
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = bboxes1[:,0], bboxes1[:,1], bboxes1[:,2], bboxes1[:,3]
    x21, y21, x22, y22 = bboxes2[:,0], bboxes2[:,1], bboxes2[:,2], bboxes2[:,3]


    x11 = x11[:,None]; y11 = y11[:,None]; x12=x12[:,None]; y12=y12[:,None]
    x21 = x21[:,None]; y21 = y21[:,None]; x22=x22[:,None]; y22=y22[:,None]

    xI1 = np.maximum(x11, np.transpose(x21))
    xI2 = np.minimum(x12, np.transpose(x22))

    yI1 = np.maximum(y11, np.transpose(y21))
    yI2 = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum((xI2 - xI1), 0.) * np.maximum((yI2 - yI1), 0.)

    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

    union = (bboxes1_area + np.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    return inter_area / (union+0.0001)


def bbox_overlap_prop_iou_corner_xy(bboxes1, bboxes2, which=1):
    """
    computes the distance matrix between two sets of bounding boxes.
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = bboxes1[:,0], bboxes1[:,1], bboxes1[:,2], bboxes1[:,3]
    x21, y21, x22, y22 = bboxes2[:,0], bboxes2[:,1], bboxes2[:,2], bboxes2[:,3]


    x11 = x11[:,None]; y11 = y11[:,None]; x12=x12[:,None]; y12=y12[:,None]
    x21 = x21[:,None]; y21 = y21[:,None]; x22=x22[:,None]; y22=y22[:,None]

    xI1 = np.maximum(x11, np.transpose(x21))
    xI2 = np.minimum(x12, np.transpose(x22))

    yI1 = np.maximum(y11, np.transpose(y21))
    yI2 = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum((xI2 - xI1), 0.) * np.maximum((yI2 - yI1), 0.) # this is the overlap. 

    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)

#    union = (bboxes1_area + np.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    if which == 1:
        return inter_area / (bboxes1_area+0.0001)
    if which == 2:
        return inter_area / (np.transpose(bboxes2_area)+0.0001)


def filter_box_mask_centroids(bbox_first_frame, mask_centroids, seg_mask, iou_thresh=0.25, bound_tol=5, max_cent_dist=50, aspect_ratio_filter=2.):

    from skimage.measure import label, regionprops
    import skimage.morphology as skmorph
    
    filtered_boxes = []
    filtered_boxes_multi_centroids = []
    filtered_boxes_multi_centroids_scores = []
    filtered_boxes_no_centroids = []
    
    filtered_boxes_centroid_assign = []
    filtered_boxes_centroid_assign_overlap = []
    
    assigned_centroids = []
    assigned_centroids_multi = []
#    unassigned_centroids = []
#    unassigned_masks = [] 
            
    for bbox_i, bbox in enumerate(bbox_first_frame):
        x1, y1, x2, y2 = bbox
        
        # instead of this we derive a disk...... -> at the centroid of the box!. 
        x1 = int(x1) #+ bound_tol
        y1 = int(y1) #+ bound_tol
        x2 = int(x2) #- bound_tol
        y2 = int(y2) #- bound_tol
        
        w = x2 - x1
        h = y2 - y1
        
        aspect_ratio = w/float(h)
        if aspect_ratio<1:
            aspect_ratio = 1./aspect_ratio
            
        # check the aspect ratio for starters! 
        if aspect_ratio <= aspect_ratio_filter:
        
            r = np.minimum(w,h) / 2.
            center = np.hstack([.5*(y1+y2),
                                .5*(x1+x2)])
            select_centroid_square = np.logical_and(np.logical_and(mask_centroids[:,0] >y1, mask_centroids[:,0] <y2), 
                                                    np.logical_and(mask_centroids[:,1] >x1, mask_centroids[:,1] <x2))
            select_centroid_circle = np.linalg.norm(mask_centroids - center[None,:], axis=-1) <= r
            
    #        n_square = np.sum(select_centroid_square)
            n_circle = np.sum(select_centroid_circle)
            
            if n_circle > 0:
                select_centroid = select_centroid_circle
            else:
    #            if n_square > 0:
                select_centroid = select_centroid_square
            
            ### select_centroid is boolean, hence need a sum!) 
            
    #        print(len(select_centroid), len(mask_centroids))
            if np.sum(select_centroid) > 0:
                select_centroid_id = np.arange(len(mask_centroids))[select_centroid]
                centroids = mask_centroids[select_centroid_id]
                
    #            print(select_centroid_id)
    #            print(bbox)
    #            print(centroids)
    #            print('====')
        #        print(len(centroids))
                if len(centroids) > 1 and len(centroids) < 10:
                    # try reducing
    #                continue
    #                print('multi')
    #                print(centroids)
#    #                print('++++')
#                    centroids_reduce = pairwise_distances(centroids)
#                    
#    #                print(centroids_reduce.max())
#    #                print('++++')
#                    if centroids_reduce.max() <= max_cent_dist:
                    assigned_centroids_multi.append(select_centroid_id+1)
                    filtered_boxes_multi_centroids.append(bbox)
                    
                    
                    # score the region ..... based on its precision ....
                    reg_mask = seg_mask.copy() >= 0
#                    reg_mask_area = np.sum(reg_mask)
                    overlap_area = np.sum(reg_mask[y1:y2, x1:x2])
                    
        #            overlap = overlap_area / float((y2-y1) * (x2-x1) + 1e-8)
    #                overlap = overlap_area / (float((y2-y1) * (x2-x1) + 1e-8) + reg_mask_area - overlap_area)
    #                overlap = overlap_area / (float(reg_mask_area + 1e-8) ) # pick the box that maximally covers target region. 
                    
                    # use the precision recall as the measure i.e the F-score. 
#                    prec = overlap_area / (float((y2-y1) * (x2-x1) + 1e-8)) 
                    prec = overlap_area
#                    filtered_boxes_multi_centroids_scores.append(float((y2-y1) * (x2-x1))) # rank by size. 
                    filtered_boxes_multi_centroids_scores.append(prec) # rank by size. 
                    
                elif  len(centroids) >= 10:
#                    assigned_centroids_multi.append(select_centroid_id)
                    continue
                else:
                    filtered_boxes.append(bbox)
                    filtered_boxes_centroid_assign.append(select_centroid_id+1)
                    
                    reg_mask = seg_mask == select_centroid_id + 1
                    
                    reg_mask_area = np.sum(reg_mask)
                    overlap_area = np.sum(reg_mask[y1:y2, x1:x2])
                    
        #            overlap = overlap_area / float((y2-y1) * (x2-x1) + 1e-8)
    #                overlap = overlap_area / (float((y2-y1) * (x2-x1) + 1e-8) + reg_mask_area - overlap_area)
    #                overlap = overlap_area / (float(reg_mask_area + 1e-8) ) # pick the box that maximally covers target region. 
                    
                    # use the precision recall as the measure i.e the F-score. 
                    prec = overlap_area / (float((y2-y1) * (x2-x1) + 1e-8)) 
                    rec = overlap_area / (float(reg_mask_area + 1e-8) )
                    
    #                overlap = 2*(prec*rec)/ (prec + rec+1e-8)
                    overlap = 0.6*prec + 0.4*rec
                    filtered_boxes_centroid_assign_overlap.append(overlap)
                    
                    assigned_centroids.append(select_centroid_id+1)
            else:
                filtered_boxes_no_centroids.append(bbox)
            
#    print('filtering duplicates')
    if len(filtered_boxes):
        filtered_boxes = np.vstack(filtered_boxes)
        
        """
        Check for more than one assignments. 
        """
        # tabulate the number that is assigned. 
        filtered_boxes_centroid_assign = np.hstack(filtered_boxes_centroid_assign)
        filtered_boxes_centroid_assign_overlap = np.hstack(filtered_boxes_centroid_assign_overlap)
        
        assigned_centroids = np.unique(assigned_centroids)
        assigned_centroids_counts = np.zeros(len(assigned_centroids), dtype=np.int)
        
        for ii in range(len(assigned_centroids_counts)):
            c_id = assigned_centroids[ii]
            assigned_centroids_counts[ii] = np.sum(filtered_boxes_centroid_assign == c_id)
        
        if np.sum(assigned_centroids_counts > 1) > 0:
            
            
            filtered_boxes_new = []
            duplicated_ids = assigned_centroids[assigned_centroids_counts > 1]
            
            for ii in range(len(filtered_boxes_centroid_assign)):
                id_ = filtered_boxes_centroid_assign[ii]
                if id_ not in duplicated_ids:
                    filtered_boxes_new.append(filtered_boxes[ii])
                    
            filtered_boxes_new = np.vstack(filtered_boxes_new)
            
            for dup_id in duplicated_ids:
#                print(dup_id)
                dup_select = filtered_boxes_centroid_assign == dup_id
                dup_select_id = np.arange(len(filtered_boxes_centroid_assign))[dup_select]
                overlap_select = filtered_boxes_centroid_assign_overlap[dup_select_id]
                
#                print(overlap_select)
                overlap_select_sort = np.sort(overlap_select)[::-1]
                
                
#                # first check whether we can separate via watershed the regions? creating regions that are not the same as currrent ids (they are not because those didn't get capture here. )
#                plt.figure()
#                plt.subplot(121)
#                plt.imshow(seg_mask == dup_id)
#                plt.subplot(122)
#                plt.imshow(watershed_cells_skimage(seg_mask == dup_id, thresh=None))
#                plt.show()
                
#                max_ratio = overlap_select_sort[0] / float(overlap_select_sort[1])
                if overlap_select_sort[0]  - overlap_select_sort[1]  >= 0.2:
                    # take only the largest.?
                    max_overlap_id = dup_select_id[np.argmax(overlap_select)]
                    filtered_boxes_new = np.vstack([filtered_boxes_new, 
                                                    filtered_boxes[max_overlap_id]])
            
            
#                    boxes = filtered_boxes[dup_select_id]
                    
#                    fig, ax = plt.subplots()
#                    ax.imshow(seg_mask)
#                    for box in boxes:
#                        x1,y1,x2,y2 = box
#                        ax.plot([x1,x2,x2,x1,x1], 
#                                [y1,y1,y2,y2,y1], lw=3)
#                    plt.show()
            
                else:
                    # check how much overlap there is between boxes. 
                    # non max suppression by overlap... ?
#                    print('non max suppression')
                    conf_boxes, _ = nms_bbox(filtered_boxes[dup_select_id], overlap_select, iou_thresh)
                    filtered_boxes_new = np.vstack([filtered_boxes_new, 
                                                    conf_boxes])
                    
#                print(filtered_boxes_centroid_assign[dup_select])
#                print(filtered_boxes_centroid_assign_overlap[dup_select])
#                print('====')
                
            filtered_boxes = filtered_boxes_new.copy()
            
#        
#    print(filtered_boxes_centroid_assign)
#    print(filtered_boxes_centroid_assign_overlap)
#    print('----')
#    print(assigned_centroids)
#    print(assigned_centroids_counts)
            
#    print(assigned_centroids_multi)
#    print(filtered_boxes_multi_centroids)
    
    if len(filtered_boxes_no_centroids) > 0:
        
        # sort the boxes by aspect ratio! first ?
        # iterate over each one and double check to see if they overlap with filtered boxes. 
        for bbox in filtered_boxes_no_centroids:
            
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            aspect_ratio = w/float(h)
            if aspect_ratio<1:
                aspect_ratio = 1./aspect_ratio
                
            if aspect_ratio <= aspect_ratio_filter:
                iou_boxes = bbox_iou_corner_xy(bbox[None,:], filtered_boxes)
                
                if iou_boxes.max() <= .25: # 0.15
                    filtered_boxes = np.vstack([filtered_boxes, bbox])

#     both of these then need to consider coverage !. 
    
    if len(filtered_boxes_multi_centroids) > 0:
        
        print(filtered_boxes_multi_centroids_scores)
        # sort the boxes by score first? 
        filtered_boxes_multi_centroids_ = [ filtered_boxes_multi_centroids[ind]  for ind in np.argsort(filtered_boxes_multi_centroids_scores)[::-1]  ]
        filtered_boxes_multi_centroids = list(filtered_boxes_multi_centroids_)
        
        print(filtered_boxes_multi_centroids)
        
        
#        fig, ax = plt.subplots()
#        ax.imshow(seg_mask)
        
#         consider coverage with existing bboxes. -> quick iou mask coverage test... 
        for bbox in filtered_boxes_multi_centroids: 
            
            # check boxes are not abnormally long. 
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            x1,y1,x2,y2 = bbox
            
#            ax.plot([x1,x2,x2,x1,x1],
#                    [y1,y1,y2,y2,y1], 
#                    lw=3)
            
#        plt.show()
            
            aspect_ratio = w/float(h)
            if aspect_ratio<1:
                aspect_ratio = 1./aspect_ratio
                
            if aspect_ratio <= aspect_ratio_filter:
            
                iou_boxes = bbox_overlap_prop_iou_corner_xy(bbox[None,:], filtered_boxes)
                
                print(bbox, iou_boxes.max())
                
                if iou_boxes.max() <= .2:
                    filtered_boxes = np.vstack([filtered_boxes, bbox])
                    
        plt.show()
#            filtered_boxes = np.vstack([filtered_boxes, filtered_boxes_multi_centroids])
        
    """
    take the largest box if they overlap fully with one of the other boxes. 
    """
    # check the unassigned centroids. (to capture those missed by bbox detections)
    keep_boxes_id = []
    overlapped_box_ids = []
    
    for bbox_i, bbox in enumerate(filtered_boxes):
        
        if bbox_i not in overlapped_box_ids:
#            print(bbox.shape)
            iou_boxes = bbox_overlap_prop_iou_corner_xy(bbox[None,:], filtered_boxes, which=2)
            
            # check if there is any boxes it fully overlaps.
            fully_overlapped_box_ids = np.arange(len(filtered_boxes))[iou_boxes.ravel() >= 0.9]
            fully_overlapped_box_ids = np.setdiff1d(fully_overlapped_box_ids,bbox_i)
            
            if len(fully_overlapped_box_ids) == 0:
                keep_boxes_id.append(bbox_i)
            else:
                # the largest is kept.
                combined_ids = np.hstack([bbox_i, fully_overlapped_box_ids])
                box_select = filtered_boxes[combined_ids]
                box_select_areas = (box_select[:,2] - box_select[:,0]) * (box_select[:,3] - box_select[:,1])
                
                max_box_id = combined_ids[np.argmax(box_select_areas)]
                keep_boxes_id.append(max_box_id)
                
                leave_box_ids = np.setdiff1d(combined_ids, max_box_id)
                overlapped_box_ids = np.hstack([overlapped_box_ids, leave_box_ids])

    keep_boxes_id = np.hstack(keep_boxes_id)
#    print(keep_boxes_id)
#        print(fully_overlapped_box_ids)
    filtered_boxes = filtered_boxes[keep_boxes_id]
    
    """
    take the remaining centroid areas that didn't have overlap boxes -> apply some operations to merge 
    """
    if len(assigned_centroids_multi) > 0:
        assigned_centroids_multi = list(np.hstack(assigned_centroids_multi))
        
    assigned_centroids = np.unique(assigned_centroids)
    print(assigned_centroids_multi)
    print(assigned_centroids)
    all_assigned_centroids = list(assigned_centroids) + assigned_centroids_multi
    all_assigned_centroids = np.unique(all_assigned_centroids)
    
    all_unique_ids = np.setdiff1d(np.unique(seg_mask), 0) # these map to the selected_ids.
    remaining_centroids_ids = np.setdiff1d(all_unique_ids, 
                                           all_assigned_centroids)
    
    if len(remaining_centroids_ids) > 0:
        
        centroid_select = np.zeros(len(all_unique_ids), dtype=np.bool)
        for c_ii, c_id in enumerate(all_unique_ids):
            if c_id in remaining_centroids_ids:
                centroid_select[c_ii] = 1
                
                
        remaining_binary = np.zeros(seg_mask.shape, dtype=np.bool)
        for rr in remaining_centroids_ids:
            remaining_binary[seg_mask==rr] = 1
        remaining_binary = skmorph.binary_dilation(remaining_binary, skmorph.disk(3))
        
        
        """
        relabel, compute eccentricity and get the proposed bounding boxes. 
        """
        remaining_labelled = label(remaining_binary)
        remaining_statsprops = regionprops(remaining_labelled)
        
        remaining_bbox = []
        remaining_ecc = []
        
        for re in remaining_statsprops:
            remaining_ecc.append(re.eccentricity) # 0 = circle. 
            remaining_bbox.append(re.bbox)
            
        remaining_bbox = np.vstack(remaining_bbox)
        remaining_ecc = np.hstack(remaining_ecc)
        
#        print(remaining_ecc)
        ecc_select = remaining_ecc<=0.95 # raise this for Xiao's as they are highly irregular.
        
#        fig, ax = plt.subplots(figsize=(10,10))
##        ax.imshow(seg_mask)
#        ax.imshow(remaining_binary)
#        print(np.sum(ecc_select))
        
        if np.sum(ecc_select) > 0:
#            print('remainder')
            for bbox in remaining_bbox[ecc_select]:
                y1,x1,y2,x2 = bbox
#                ax.plot([x1,x2,x2,x1,x1], 
#                        [y1,y1,y2,y2,y1], lw=3)
            
                filtered_boxes = np.vstack([filtered_boxes, [x1,y1,x2,y2]])
#        ax.plot(mask_centroids[centroid_select, 1], 
#                mask_centroids[centroid_select, 0], 'go')
#        plt.show()
        
    return filtered_boxes


"""
bunch of scripts to help single organoid tracking. 
"""
def unique_pts(a):

    return np.vstack(list({tuple(row) for row in a}))


def delaunay_region_pts_nofilt_fixed_dist(pts, shape, outlier_thresh_dist=30, edge_filter=None ):
    
    from scipy.spatial import Delaunay
    from skimage.draw import polygon, line_aa
    from skimage.draw import line_aa
    import networkx as nx
    from sklearn.metrics.pairwise import pairwise_distances
#>>> img = np.zeros((10, 10), dtype=np.uint8)
#>>> rr, cc, val = line_aa(1, 1, 8, 8)
#>>> img[rr, cc] = val * 255
    
#    all_delaunay_lengths = []
    
    mask = np.zeros(shape, dtype=np.bool)
    
#    if len(select) > 0:
        
#        pts_dist_select = pts_dist[select].sum(axis=0) > 5
##        select = 
#        pts = pts[np.hstack(select)][pts_dist_select]
#        print(len(pts))
        
    """ DELAUNAY """
    tri = Delaunay(pts) 
    
#    for tri_pts in tri.simplices:
#        
#        tri_pts = pts[tri_pts]
#        
#        tri_lengths = np.hstack([np.linalg.norm(tri_pts[0] - tri_pts[1]), 
#                                 np.linalg.norm(tri_pts[1] - tri_pts[2]),
#                                 np.linalg.norm(tri_pts[2] - tri_pts[0])])
#        all_delaunay_lengths.append(np.nanmedian(tri_lengths))
#        
#    all_delaunay_lengths = np.hstack(all_delaunay_lengths)
#    
#    mean_dist = np.nanmean(all_delaunay_lengths)
#    std_dist = np.nanstd(all_delaunay_lengths)
    
#    if outlier_thresh is not None:
    outlier_length = outlier_thresh_dist
    
        
    for tri_pts in tri.simplices:
        
        tri_pts = pts[tri_pts]
        
        tri_lengths = np.hstack([np.linalg.norm(tri_pts[0] - tri_pts[1]), 
                                 np.linalg.norm(tri_pts[1] - tri_pts[2]),
                                 np.linalg.norm(tri_pts[2] - tri_pts[0])])
            
#        if outlier_thresh is None:
#            
#            rr, cc = polygon(tri_pts[:,0], tri_pts[:,1])
#            mask[rr,cc] = 1
#        else:
    #            tri_lengths_test = tri_lengths>= np.maximum(50., outlier_length)
        tri_lengths_test = tri_lengths > outlier_length
        keep_triangle = np.sum(tri_lengths_test) == 0
#        keep_triangle = 1
        
        # check kedges. 
        if edge_filter is not None:
#            line1 = draw_line_mask(tri_pts[0], tri_pts[1], edge_filter.shape)
#            line2 = draw_line_mask(tri_pts[1], tri_pts[2], edge_filter.shape)
#            line3 = draw_line_mask(tri_pts[2], tri_pts[0], edge_filter.shape)
            
            line1 = edge_filter[np.linspace(tri_pts[0,0], tri_pts[1,0], 3*tri_lengths[0]).astype(np.int), 
                                np.linspace(tri_pts[0,1], tri_pts[1,1], 3*tri_lengths[0]).astype(np.int)]
            line2 = edge_filter[np.linspace(tri_pts[1,0], tri_pts[2,0], 3*tri_lengths[1]).astype(np.int), 
                                np.linspace(tri_pts[1,1], tri_pts[2,1], 3*tri_lengths[1]).astype(np.int)]
            line3 = edge_filter[np.linspace(tri_pts[2,0], tri_pts[0,0], 3*tri_lengths[2]).astype(np.int), 
                                np.linspace(tri_pts[2,1], tri_pts[0,1], 3*tri_lengths[2]).astype(np.int)]
            
            line_keep = np.max(edge_filter[line1]) + np.max(edge_filter[line2]) + np.max(edge_filter[line3])
            line_keep = line_keep == 0
            keep_triangle = np.logical_and(keep_triangle, line_keep)
            
        if keep_triangle == True:
            # check the length distribution -> draw only if they are dont involve non extremal lengths. 
            rr, cc = polygon(tri_pts[:,0], tri_pts[:,1])
            mask[rr,cc] = 1
#    else:
#        print('empty select')
    return mask>0

def motion_refine_segmentation_tracks( tracks, segs, im_shape, spixel_size, initial_dist_thresh=1.2, initial_region_min_size=100, alpha_dist_thresh=None, min_motion_frac=0.4 , debug_viz=False):
    
    """
    tracks is the meantracks (dense or not)
    segs is the list of segmentation seeds.
    """
    import numpy as np 
    from skimage.filters import threshold_otsu
    import skimage.measure as skmeasure
    import skimage.morphology as skmorph
    
    # compute the source map. 
#    spixel_size = np.abs(tracks[1,-1,1] - tracks[0,0,1])
#    
#    print(spixel_size)
        
    # filter and grab only the most moving tracks.
    move_spixels = np.sum(np.linalg.norm(tracks[:,1:] - tracks[:,:-1], axis=-1), axis=1) # cumulative mode.
#    thresh_move = threshold_otsu(move_spixels)
    thresh_move = np.mean(move_spixels)
    move_spixels_select = move_spixels>= thresh_move # global pixel-ids that move. 
    
    """
    build the region which these points map out in the initial frame. 
    """
    source_pts = tracks[move_spixels_select, 0]
    source_pts = unique_pts(source_pts)
    
    
    source_binary = delaunay_region_pts_nofilt_fixed_dist(source_pts, 
                                                            im_shape, 
                                                            outlier_thresh_dist=initial_dist_thresh*spixel_size, 
                                                            edge_filter=None )
    
    # clean up by removing overly small size.
    source_binary = skmorph.remove_small_objects(source_binary, min_size=initial_region_min_size)
    
    source_label = skmeasure.label(source_binary) # connected component analysis of spatial points. 
    
    if debug_viz:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('source_map')
        ax.imshow(source_label)
        plt.show()
        
    
    print('attempting assigning tracks to source ... ')
    """
    Source - segmentation assignment filtering. 
    """
    segs_source_assign = [np.unique(np.setdiff1d(source_label[seg>0], 0)) for seg in segs] # how to handle multiple motion assignments. 
    # this is the score of the motion coverage. 
    segs_source_coverage = np.hstack([np.mean(source_binary[seg>0]) for seg in segs]) # how much of the body covers a motion source. 
        
    
    print(segs_source_assign)
    print(segs_source_coverage)
    print(len(segs_source_assign))
    
    """
    Main part of algorithm, constructing the segments assigned to each region. 
    1) assign if unique to clique. ( with possibly an absolute maximum range? )
    2) resolve ambiguities. 
    """
    all_mapped_source_ids = np.unique(np.hstack([aa for aa in segs_source_assign if len(aa)>0]))
    all_assoc_reg_ids = [[] for ii in range(len(all_mapped_source_ids))]
    for source_id_ii, source_id in enumerate(all_mapped_source_ids):
        
        for ii in range(len(segs_source_assign)):
            if source_id in segs_source_assign[ii]:
                all_assoc_reg_ids[source_id_ii].append(ii)
    
    num_reg_ids_sources = np.hstack([len(ss) for ss in all_assoc_reg_ids])
    
    uniq_source_ids = all_mapped_source_ids[num_reg_ids_sources == 1]
    uniq_source_reg_ids = np.hstack([all_assoc_reg_ids[ss] for ss in np.arange(len(all_assoc_reg_ids))[num_reg_ids_sources == 1]])
    nonuniq_source_ids = all_mapped_source_ids[num_reg_ids_sources > 1]
    
    """
    Now we start resolving and assigning...
    easier to iterate over the unique.... 
    
    
    # this is failing utterly miserably at the moment for motion source id refinement. 
    """
    segs_spixels_source_assign = []
    associated_segs = []
    parsed_segs = []
    
    for ii in range(len(all_mapped_source_ids)):
        
        source_id = all_mapped_source_ids[ii]
        reg_id = all_assoc_reg_ids[ii]
        
        if source_id in uniq_source_ids:
            
            source_mask = source_label==source_id
            spixels_source_select = source_mask[tracks[:,0,0], 
                                                tracks[:,0,1]] > 0
                                            
            pts_mask = tracks[spixels_source_select,0]
            
            if alpha_dist_thresh is None:
                pts_mask_convex = delaunay_region_pts_nofilt_fixed_dist(pts_mask, 
                                                                        im_shape, 
                                                                        outlier_thresh_dist=2*np.max(im_shape), 
                                                                        edge_filter=None )
            else:
                pts_mask_convex = delaunay_region_pts_nofilt_fixed_dist(pts_mask, 
                                                                        im_shape, 
                                                                        outlier_thresh_dist=alpha_dist_thresh * spixel_size, 
                                                                        edge_filter=None )
            
            spixels_refine_select = pts_mask_convex[tracks[:,0,0], 
                                                    tracks[:,0,1]] > 0
            spixels_refine_select = np.arange(len(tracks))[spixels_refine_select]
                                                    
            segs_spixels_source_assign.append(spixels_refine_select)
            
#            print(reg_id)
            associated_segs.append(reg_id[0])
            parsed_segs.append(reg_id[0])
            
            if debug_viz:
                plt.figure()
                plt.imshow(pts_mask_convex)
                plt.plot(tracks[spixels_refine_select,0,1], 
                         tracks[spixels_refine_select,0,0], '.')
                plt.show()
        else:
            
            # resolve between two different ids., 
            source_id = all_mapped_source_ids[ii]
            reg_id_map = all_assoc_reg_ids[ii]
            
            # partition the clique amongst the regions. -> if a region is part of a previously unique region then don't do anything?
            """
            split the region to the other ids and add to the list of tracks. 
            """
#            print(source_id)
#            print(reg_id_map)
#                reg_id_map_parse = np.setdiff1d(reg_id_map, uniq_source_reg_ids)
            reg_id_map_parse = list(reg_id_map)
            
#            print('splitting ... ')
            source_mask = source_label==source_id
            spixels_source_select = source_mask[tracks[:,0,0], 
                                                tracks[:,0,1]] > 0
            
            spixels_source_select_pts = tracks[spixels_source_select,-1]
                                                
            seg_pts = [extract_boundary_binary(segs[idd]) for idd in reg_id_map_parse]
            
            spixels_source_seg_dist = [dist_pt_set(ss_pts, spixels_source_select_pts) for ss_pts in seg_pts]
            spixels_source_seg_dist = np.vstack(spixels_source_seg_dist)
            spixels_source_seg_dist_assign = np.argmin(spixels_source_seg_dist, axis=0)
            
            
            if debug_viz:
                fig, ax = plt.subplots()
                ax.imshow(img[0])
                ax.plot(spixels_source_select_pts[:,1], 
                        spixels_source_select_pts[:,0], 'g.', alpha=0.5)
                for ss in seg_pts:
                    ax.plot(ss[:,1], 
                            ss[:,0], color='r', lw=3)
                plt.show()
                
                fig, ax = plt.subplots()
                ax.imshow(img[0])
    #                    ax.plot(spixels_source_select_pts[:,1], 
    #                            spixels_source_select_pts[:,0], 'g.', alpha=0.5)
                
                for uu in np.unique(spixels_source_seg_dist_assign):
                    ax.plot(spixels_source_select_pts[spixels_source_seg_dist_assign==uu,1], 
                            spixels_source_select_pts[spixels_source_seg_dist_assign==uu,0], '.', alpha=0.5)
                for ss in seg_pts:
                    ax.plot(ss[:,1], 
                            ss[:,0], color='r', lw=3)
                plt.show()
            
            
            # iterate over the assigned and add to the parsed out array.  
            for re_ii, re in enumerate(reg_id_map_parse):
                spixels_assign_re = spixels_source_seg_dist_assign == re_ii
                if np.sum(spixels_assign_re) > 3:
                    pts_assign = spixels_source_select_pts[spixels_assign_re]
                    
                    if alpha_dist_thresh is None:
                        pts_mask_convex = delaunay_region_pts_nofilt_fixed_dist(pts_assign, 
                                                                                im_shape, 
                                                                                outlier_thresh_dist=2*np.max(im_shape), 
                                                                                edge_filter=None )
                    else:
                        pts_mask_convex = delaunay_region_pts_nofilt_fixed_dist(pts_assign, 
                                                                                im_shape, 
                                                                                outlier_thresh_dist=alpha_dist_thresh*spixel_size, 
                                                                                edge_filter=None )
                    if debug_viz:
                        plt.figure()
                        plt.imshow(pts_mask_convex)
                        plt.show()
            
                    spixels_refine_select = pts_mask_convex[tracks[:,0,0], 
                                                            tracks[:,0,1]] > 0
                    spixels_refine_select = np.arange(len(tracks))[spixels_refine_select]
                                                            
                    segs_spixels_source_assign.append(spixels_refine_select)
                    associated_segs.append(re)
                    parsed_segs.append(re)
                    
    print(associated_segs)
    
    """
    iterate over the regions again. 
    """
    spixels_final_motion_assign = []
    
    for reg_id in np.arange(len(segs)):
        motion_score = segs_source_coverage[reg_id]
        print(reg_id, motion_score)
        
        if motion_score >= min_motion_frac:
            # iterate over the motion suggested tracks and combine.
            all_reg_id_source_assign = []
            
            for ss_i in range(len(associated_segs)):
                if associated_segs[ss_i] == reg_id: 
                    segs_spixels = segs_spixels_source_assign[ss_i]
                    if len(segs_spixels) > 0:
                        all_reg_id_source_assign.append(segs_spixels)
                        
            if len(all_reg_id_source_assign) > 0:
                all_reg_id_source_assign = np.unique(np.hstack(all_reg_id_source_assign))
            
            spixels_final_motion_assign.append(all_reg_id_source_assign)
        else:
            spixels_final_motion_assign.append([])
    
    return spixels_final_motion_assign, source_label


def extract_boundary_binary(binary):
    
    from skimage.measure import find_contours
    
    cont = find_contours(binary>0, level=0)
    cont_len = [len(cc) for cc in cont]
    
    return cont[np.argmax(cont_len)]
    
def dist_pt_set(ref_pts1, query_pts2):
    
    from sklearn.metrics.pairwise import pairwise_distances
    
    dist_matrix = pairwise_distances(query_pts2, ref_pts1 )
    
    return np.min(dist_matrix,axis=1)

def PCA_embed_track_model(tracks, n_components=3, whiten=True):
    
    from sklearn.decomposition import PCA
    
    #==============================================================================
    #   Train a PCA model with all tracks          
    #==============================================================================        
    pca_model = PCA(n_components=n_components, whiten=whiten)
    pca_model.fit_transform(np.hstack([tracks[:,:,1], tracks[:,:,0]]))
    
    return pca_model


def compute_segs_from_binary_and_bbox(bboxes, mask_binary, dilate=None, dilate_mask=None, fill_holes=True):
    
    import skimage.morphology as skmorph
    from scipy.ndimage.morphology import binary_fill_holes
    
    nrows, ncols = mask_binary.shape[:2]
    
    # grab a list of binary regions for each individual organoid. 
    segs = []
    
    for bbox in bboxes:
        
        x1,y1,x2,y2 = bbox; 
        x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
        
        # make sure to clip these.
        x1 = np.clip(x1, 0, ncols-1)
        x2 = np.clip(x2, 0, ncols-1)
        y1 = np.clip(y1, 0, nrows-1)
        y2 = np.clip(y2, 0, nrows-1)
        
        seg = np.zeros_like(mask_binary)
        seg[y1:y2, x1:x2] = 1 
        
        # joint masking of the region. 
        seg = np.logical_and(seg, mask_binary)
        
        if fill_holes:
            seg = binary_fill_holes(seg)
        
        if dilate is not None:
            seg = skmorph.binary_dilation(seg, dilate_mask(dilate))
            
        segs.append(seg)
        
    return segs

def mkdir(directory):
    
    import os 
#    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return []



def get_centroids_and_area(labelled):
    
    from skimage.measure import regionprops
    
    regions = regionprops(labelled)
    
    region_areas = np.hstack([re.area for re in regions])
    region_centroids = np.vstack([re.centroid for re in regions])
    
    return region_centroids, region_areas

def filter_min_max_areas(cells, min_thresh=0, max_thresh=np.inf): 
    
    cells_ = cells.copy()
    mask_centroids, mask_areas = get_centroids_and_area(cells)
    
    min_cell_area_filter = mask_areas < min_thresh
    max_cell_area_filter = mask_areas> max_thresh
    
    if min_cell_area_filter.sum() > 0:
        min_cell_area_filter_ids = np.setdiff1d(np.unique(cells), 0)[min_cell_area_filter]
        
        for iid in min_cell_area_filter_ids:
            cells_[cells==iid] = 0
    
    if max_cell_area_filter.sum() > 0:
        max_cell_area_filter_ids = np.setdiff1d(np.unique(cells), 0)[max_cell_area_filter]
        
        for iid in max_cell_area_filter_ids:
            cells_[cells==iid] = 0
    
    return cells_


def iou_box_voc(box_1, box_2):
    
    x1_tl, y1_tl, x1_br, y1_br = box_1[1:]
    x2_tl, y2_tl, x2_br, y2_br = box_2[1:]
#    x2_tl = detection_2[1]
#    x1_br = detection_1[0] + detection_1[3]
#    x2_br = detection_2[0] + detection_2[3]
#    y1_tl = detection_1[1]
#    y2_tl = detection_2[1]
#    y1_br = detection_1[1] + detection_1[4]
#    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    
    w1 = x1_br - x1_tl
    h1 = y1_br - y1_tl
    w2 = x2_br - x2_tl
    h2 = y2_br - y2_tl
    
    overlap_area = x_overlap * y_overlap
    area_1 = w1*h1
    area_2 = w2*h2
    
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


def nonmax_suppress_boxes(boxes, iou_thresh=0.1, threshold=0.5):
    
    detections = sorted(boxes, key=lambda x: x[0],
            reverse=True) # sorted in terms of probability !. 
    
    new_detections=[]
    
    new_detections.append(detections[0])
    del detections[0]
    
    for index, detection in enumerate(detections):
        
#        print(detections)
        if detection[0] > threshold:
            # its probability is above the threshold. 
            
            for new_detection in new_detections:
                iou_box = iou_box_voc(detection, new_detection)
#                print(index, iou_box, iou_thresh)
                
                # both coverage and detection must check out. 
                if iou_box > iou_thresh: # overlap this a lot. 
                    del detections[index]
                    break
            else:
                new_detections.append(detection)
                del detections[index]
            
    if len(new_detections)>0:
        new_detections = np.vstack(new_detections)
            
    return new_detections


def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)#[::-1]

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last] # this is a LIFO system. ! 
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def nonmax_suppress_boxes_with_binary(boxes, mask, iou_thresh=0.5, threshold=0.1, use_prob=False):
    
    m, n = mask.shape[:2]
    
    
    if use_prob:
    
        boxes_ = boxes.copy()
    else:
        probs = []
    
        for box in boxes:
            
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            prob = np.mean(mask[y1:y2,x1:x2])
            probs.append(prob)
        
        if len(probs) > 0:
            probs = np.hstack(probs)
            boxes_ = np.hstack([probs[:,None], 
                                boxes])

#    print(boxes_)
#    print('====')
    if len(boxes_) > 0: 
        # this is a big problem. 
#        boxes_ = nonmax_suppress_boxes(boxes_, 
#                                       iou_thresh=iou_thresh, 
#                                       threshold=threshold)
        boxes_ = boxes_[boxes_[:,0]>threshold]
        if len(boxes_) > 0:
            boxes_ = non_max_suppression_fast(boxes_[:,1:], 
                                              probs=boxes_[:,0], 
                                              overlapThresh=iou_thresh)
    else:    
        boxes_ = boxes
    return boxes_


def remove_very_large_bbox(boxes, shape, thresh=0.5):
    
    areas_box = []
    area_shape = float(np.prod(shape))
    
#    print(area_shape)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        
        area_box = (y2-y1)*(x2-x1)
        areas_box.append(area_box)
        
    areas_box = np.hstack(areas_box)
    areas_box_frac = areas_box / float(area_shape)
#    print(areas_box_frac)
    return boxes[areas_box_frac<=thresh]

def read_video_cv2(avifile):
    
    import cv2
    
    vidcap = cv2.VideoCapture(avifile)
    success,image = vidcap.read()
    
    vid_array = []
    
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if success:
            vid_array.append(image)
        count += 1
        
    vid_array = np.array(vid_array)
      
    return vid_array


def compile_boundaries_arrays(expt, boundaries, total_vid_frames):
    
    """
    see below, this is done in a similar manner just to match...... the one below.... so in analysis we can co-plot the boundary variation. 
    """
   
    filenames_all = []
    conditions_all = []
    genetics_all = []
    img_channel_no_all = []
    org_id_all = []
    frame_no_all = []
    pixel_resolution_all = []
    frame_duration_all = []
    total_vid_frames_all = []
    
    boundaries_all_export = [] 
    
    for channel_ii, boundaries_channel in enumerate(boundaries):
        # -> iterate over channels.
#        print(boundaries_channel.shape)
        
        for bb_ii, boundaries_channel_ii in enumerate(boundaries_channel):
            # -> iterate over timepoints. 
#                print(boundaries_channel_ii.shape)
            # bb_ii is the org_id. 
            for bb_ii_tt, boundaries_ii_tt in enumerate(boundaries_channel_ii):
#                    print(len(metrics_names))
                # if features are nan do nothing
#                    print(boundaries_ii_tt.shape) # this can also be empty!.
                # these are now coordinates.... 
#                    if len(boundaries_ii_tt) > 0:
                if ~np.isnan(boundaries_ii_tt[0][0]):
                    
                    filenames_all.append(expt)
                    img_channel_no_all.append(channel_ii+1)
                    org_id_all.append(bb_ii+1)
                    frame_no_all.append(bb_ii_tt+1)
                    pixel_resolution_all.append(pixel_res)
                    frame_duration_all.append(time_res)
                    total_vid_frames_all.append(total_vid_frames) # this is causing the issue
                    
#                            print(boundaries_ii_tt.shape)
                    boundaries_all_export.append(boundaries_ii_tt)
                        
    
#    print(np.array(boundaries_all_export).shape)
#    print(filenames_all)
    filenames_all = np.hstack(filenames_all)
    # conditions_all = np.hstack(conditions_all)
    # genetics_all = np.hstack(genetics_all)
    img_channel_no_all = np.hstack(img_channel_no_all)
    org_id_all = np.hstack(org_id_all)
    frame_no_all = np.hstack(frame_no_all)
    pixel_resolution_all = np.hstack(pixel_resolution_all)
    frame_duration_all = np.hstack(frame_duration_all)
    total_vid_frames_all = np.hstack(total_vid_frames_all)
#    
##    print(filenames_all.shape)
##    print(len(boundaries_all_export))
#    # the creation of the array kills? 
    boundaries_all_export = np.array(boundaries_all_export)
#    print(boundaries_all_export.shape)
    
    return (filenames_all, 
            img_channel_no_all, org_id_all, frame_no_all, 
            pixel_resolution_all, frame_duration_all, total_vid_frames) , boundaries_all_export


def construct_metrics_table_csv(expt, metrics, metricslabels, total_vid_frames):
    
    import pandas as pd 
    import numpy as np 

    all_data = []
    metrics_names = [name.strip() for name in metricslabels]
   
    for channel_ii, metrics_channel in enumerate(metrics):
            
        # do nothing otherwise.
        for bb_ii, metrics_ii in enumerate(metrics_channel):
            
            # bb_ii is the org_id. 
            for bb_ii_tt, metrics_ii_tt in enumerate(metrics_ii):
#                    print(len(metrics_names))
                # if features are nan do nothing
                
                if ~np.isnan(metrics_ii_tt[0]):
                    data = np.hstack([ expt, # filename
                                       channel_ii+1, #want the image channel.
                                       bb_ii+1,  # want the oranoid id.
                                       bb_ii_tt+1, # want the frame_no.
                                       pixel_res,
                                       time_res, 
                                       total_vid_frames, 
                                       metrics_ii_tt, 
                                        ])
#                        print(len(data))
#                        print(data[:11])
                    all_data.append(data)
                        
    all_data = np.array(all_data)
    headers = np.hstack(['Filename',
                         'Img_Channel_No', 
                         'Org_ID', 
                         'Frame_No', 
                         'pixel_resolution[um]',
                         'Frame_Duration[h]',
                         'Total_Video_Frame_No',
                         metrics_names])
                        
#    print(all_data.shape, headers.shape)
    
    all_data = pd.DataFrame(all_data, 
                            index=None,
                            columns=headers)
    
    return all_data


def load_pickle(savefile):
    import pickle
    with open(savefile, 'rb') as handle:
        b = pickle.load(handle)
        
    return b



# def construct_metrics_table_csv(expt, org_expt_table, metrics, metricslabels):
    
#     import pandas as pd 
#     import numpy as np 
    
#     # get the corresponding info for the given experiment
#     select_meta = np.arange(len(org_expt_table))[org_expt_table['Filename'].values == expt][0]
#     meta_info = org_expt_table.loc[select_meta]
    
#     # parse the genotypes and channel information.
#     expt_channels = meta_info['Img_Channel']
#     if isinstance(expt_channels,np.int64):
#         expt_channels = np.hstack([int(expt_channels)])
#     else:
#         if ',' in expt_channels:
#             expt_channels = expt_channels.split(',')
#             expt_channels = np.hstack([int(ch) for ch in expt_channels])
#         else:
#             expt_channels = np.hstack([int(expt_channels)])
        
#     genotype_channels = meta_info['Condition']
#     if '+' in genotype_channels:
#         genotype_channels = genotype_channels.split('+')
#         genotype_channels = np.hstack([ge.strip() for ge in genotype_channels])
#     else:
#         genotype_channels = np.hstack([genotype_channels.strip()])
     
#     pixel_res = meta_info['pixel_resolution[um]']
#     time_res = meta_info['time_resolution[h]']

#     total_vid_frames = meta_info['n_frames'] # not really used per say .... 
    
#     # construct a pandas table. 
# #    print(expt_channels)
# #    print(genotype_channels)
    
#     all_data = []
#     metrics_names = [name.strip() for name in metricslabels]
    
#     print(genotype_channels, expt_channels)
    
#     for channel_ii, metrics_channel in enumerate(metrics):
        
#         if (channel_ii+1) in list(expt_channels):
#             gene = genotype_channels[expt_channels==channel_ii+1][0]
            
#             # do nothing otherwise.
#             for bb_ii, metrics_ii in enumerate(metrics_channel):
                
#                 # bb_ii is the org_id. 
#                 for bb_ii_tt, metrics_ii_tt in enumerate(metrics_ii):
# #                    print(len(metrics_names))
#                     # if features are nan do nothing
#                     if ~np.isnan(metrics_ii_tt[0]):
#                         data = np.hstack([ expt, # filename
#                                            meta_info['Condition'].strip(), #experimental condition in the well.
#                                            gene, # want the genetics of this orgnaoid
#                                            channel_ii+1, #want the image channel.
#                                            bb_ii+1,  # want the oranoid id.
#                                            bb_ii_tt+1, # want the frame_no.
#                                            pixel_res,
#                                            time_res, 
#                                            total_vid_frames, 
#                                            metrics_ii_tt, 
#                                             ])
# #                        print(len(data))
# #                        print(data[:11])
#                         all_data.append(data)
                        
#     all_data = np.array(all_data)
#     headers = np.hstack(['Filename',
#                          'Condition',
#                          'Genetics',
#                          'Img_Channel_No', 
#                          'Org_ID', 
#                          'Frame_No', 
#                          'pixel_resolution[um]',
#                          'Frame_Duration[h]',
#                          'Total_Video_Frame_No',
#                          metrics_names])
                        
# #    print(all_data.shape, headers.shape)
    
#     all_data = pd.DataFrame(all_data, 
#                             index=None,
#                             columns=headers)
    
#     return all_data


def map_intensity_interp2(query_pts, grid_shape, I_ref, method='spline', cast_uint8=False, s=0):

    import numpy as np 
    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator 
    
    if method == 'spline':
        spl = RectBivariateSpline(np.arange(grid_shape[0]), 
                                  np.arange(grid_shape[1]), 
                                  I_ref,
                                  s=s)
        I_query = spl.ev(query_pts[...,0], 
                         query_pts[...,1])
    else:
        spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                       np.arange(grid_shape[1])), 
                                       I_ref, method=method, bounds_error=False, fill_value=0)
        I_query = spl((query_pts[...,0], 
                       query_pts[...,1]))

    if cast_uint8:
        I_query = np.uint8(I_query)
    
    return I_query

# =============================================================================
# Do the ALS algorithm
# =============================================================================
def baseline_als(y, lam, p, niter=10):
	r""" Estimates a baseline signal using asymmetric least squares. It can also be used for generic applications where a 1D signal requires smoothing.
	Specifically the baseline signal, :math:`z` is the solution to the following optimization problem 

	.. math::
		z = arg\,min_z \{w(y-z)^2 + \lambda\sum(\Delta z)^2\}

	where :math:`y` is the input signal, :math:`\Delta z` is the 2nd derivative, :math:`\lambda` is the smoothing regularizer and :math:`w` is an asymmetric weighting

	.. math::
		w = 
		\Biggl \lbrace 
		{ 
		p ,\text{ if } 
		  {y>z}
		\atop 
		1-p, \text{ otherwise } 
		}

	Parameters
	----------
	signal : 1D numpy array
		The 1D signal to estimate a baseline signal. 
	p :  scalar
		Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
	lam : scalar
		Controls the degree of smoothness in the baseline
	niter: int
		The number of iterations to run the algorithm. Only a few iterations is required generally. 

	Returns
	-------
	z : 1D numpy array
		the estimated 1D baseline signal

	See Also
	--------
	unwrap3D.Analysis_Functions.timeseries.baseline_correction_time : 
		Application of this method to estimate a baseline for a 1D signal and correct the signal e.g. for photobleaching
	unwrap3D.Analysis_Functions.timeseries.decompose_nonlinear_time_series :
		Application of this method to decompose a 1D signal into smooth baseline + high frequency fluctuations. 

	"""
	from scipy import sparse
	from scipy.sparse.linalg import spsolve
	import numpy as np 
	
	L = len(y)
	D = sparse.csc_matrix(np.diff(np.eye(L), 2))
	w = np.ones(L)
	for i in range(niter):
		W = sparse.spdiags(w, 0, L, L)
		Z = W + lam * D.dot(D.transpose())
		z = spsolve(Z, w*y)
		w = p * (y > z) + (1-p) * (y < z)
	return z

def get_colors(inp, colormap, vmin=None, vmax=None):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))

if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    import scipy.io as spio
    import os 
    import skimage.io as skio 
    # import MOSES.Utility_Functions.file_io as fio
    import glob
    import skimage.morphology as skmorph
    from scipy.ndimage.morphology import binary_fill_holes
#    from detect_organoid import clean_and_seg_organoids_Xiao_VC
    from skimage.restoration import denoise_nl_means
    from skimage.filters.rank import median, entropy
    import seaborn as sns
    from skimage.filters import gaussian
    from skimage.exposure import rescale_intensity
    from skimage.measure import find_contours
    from tqdm import tqdm
    
    from scipy.ndimage.measurements import center_of_mass
    import skimage.measure as skmeasure
    from skimage.util import img_as_ubyte
    from skimage.filters import threshold_otsu
    import skimage.transform as sktform 
    
    import seaborn as sns 
    import glob 
    
    from hmmviz import TransGraph

    import numpy as np 
    import glob 
    import os 
    import pylab as plt 
    from tqdm import tqdm 
    import skimage.exposure as skexposure 
    import scipy.io as spio 
    import pandas as pd 
    
    import skimage.filters as skfilters
    import skimage.transform as sktform
        
    
    rootfolder = r'C:\Users\s205272\Documents\Work\Research\Lu_Lab\SAM_paper_2023\Other_Manuscripts\Cell_Tracking_Challenge'
    saverootfolder = r'C:\Users\s205272\Documents\Work\Research\Lu_Lab\SAM_paper_2023\Other_Manuscripts\Cell_Tracking_Challenge'
    
    
    saveplotsfolder = r'C:\Users\s205272\Documents\Work\Research\Lu_Lab\SAM_paper_2023\Other_Manuscripts\Cell_Tracking_Challenge_Results'
    # saveplotsfolder = r'C:\Users\s205272\Documents\Work\Research\Lu_Lab\SAM_paper_2023\Other_Manuscripts\Cell_Tracking_Challenge_Results_PCA-init'
    mkdir(saveplotsfolder)
    
    
    # cellfolders = [r'PhC-C2DH-U373\01', #, # done  ---- done 
    #                 r'PhC-C2DH-U373\02'] #done 

    # saveplotsfolder_cells = os.path.join(saveplotsfolder, 
    #                                       '2023-10-04_U373_modules_github_test')
    # mkdir(saveplotsfolder_cells)
    
    
    cellfolders = [r'Fluo-N2DH-SIM+\01', #, # done  ---- done 
                    r'Fluo-N2DH-SIM+\02'] #done 

    saveplotsfolder_cells = os.path.join(saveplotsfolder, 
                                          # '2023-05-23_N2DH-SIM+_modules_3_plot',
                                           '2023-05-23_N2DH-SIM+_modules_github_test')
    mkdir(saveplotsfolder_cells)
    
    
# =============================================================================
#     Do the rest of the analysis now. 
# =============================================================================

    """
    Define the imports here. 
    """
    import SPOT.Utility_Functions.file_io as fio 
    import SPOT.Analysis.preprocessing as preprocess
    import SPOT.Analysis.sam_analysis as SAM
    import SPOT.Utility_Functions.plotting as plotting
    
    analysisfolders = []
    expts = []
    
    for cellfolder in tqdm(cellfolders[:]):
        infolder = os.path.join(rootfolder,cellfolder)
        rootname, basename = os.path.split(cellfolder)
        saveresultsfolder = os.path.join(saverootfolder, rootname, basename+'_SAM'); # mkdir(saveresultsfolder)
        
        expt = rootname+'_'+basename
        
        expts.append(expt)
        analysisfolders.append(saveresultsfolder)
        
    expts = ['','']
    """
    load
    """
    all_feats, metadict = fio.load_SPOT_features_files(analysisfolders, 
                                                        expts,
                                                        boundaryfile_suffix='final_boundaries.mat',
                                                        boundarykey='boundary',
                                                        patchfile_suffix='final_img_patches_noresize_filter.mat',
                                                        shapefeatsfile_suffix='final_shape_metrics_filter.csv',
                                                        appearfeatsfile_suffix='final_appearance_metrics_filter.csv', 
                                                        motionfeatsfile_suffix='final_motion_metrics_filter.csv',
                                                        read_chunksize=2000)
                                   

    """
    Do the full combine across the 2 datasets.
    """
    all_uniq_org_ids = metadict['all_object_uniq_row_ids']
    all_dataset_boundaries = metadict['all_object_boundaries']
    
    all_patches = list(metadict['all_object_patches'])    
    all_patches = np.array(all_patches)[None,...] # to make the same. 
    std_size=(64,64)
    all_patches_std_size = np.array([sktform.resize(pp, output_shape=std_size, order=1, preserve_range=True) for pp in all_patches[0]])
    all_patches_sizes = metadict['all_object_patches_sizes']
    
    # all_expts = metadict['']

# =============================================================================
# =============================================================================
# #     Remove features that shouldn't have been computed 
# =============================================================================
# =============================================================================
    feature_names = metadict['feature_names']                  
    
    """
    parse out the division 
    """
    all_division = np.squeeze(all_feats[:,feature_names=='div_bool']).copy() # this needs to  be taken out of all_feats. 

    
    # number 2 is identification of velocity vector columns. 
    mean_global_velocity_xy_cols = []
    for feat_ii, feature_name in enumerate(feature_names):
        if 'mean_disp_global_' in feature_name and 'flow' not in feature_name:
            mean_global_velocity_xy_cols.append(feat_ii)
    mean_global_velocity_xy_cols = np.hstack(mean_global_velocity_xy_cols)
            
    mean_global_flow_velocity_xy_cols = []
    for feat_ii, feature_name in enumerate(feature_names):
        if 'mean_disp_global_flow_' in feature_name:
            mean_global_flow_velocity_xy_cols.append(feat_ii)
    mean_global_flow_velocity_xy_cols = np.hstack(mean_global_flow_velocity_xy_cols)

    div_bool_cols = []
    for feat_ii, feature_name in enumerate(feature_names):
        if 'div_bool' in feature_name:
            div_bool_cols.append(feat_ii)
    div_bool_cols = np.hstack(div_bool_cols)

    select_feats_index = np.setdiff1d(np.arange(len(feature_names)), 
                                  np.hstack([mean_global_velocity_xy_cols,
                                              mean_global_flow_velocity_xy_cols,
                                              div_bool_cols]))
    
    
    
    
    # remove the vectorial information as well as cell division only information.  
    all_feats = all_feats[:,select_feats_index].copy()
    feature_names = feature_names[select_feats_index].copy()


    # =============================================================================
    #   remove threshold adjacency stats.   
    #===========================================================================        
    # area_feat_ind = np.arange(len(feature_names))[np.hstack(['area' in feat_name for feat_name in feature_names])][0]
    # drop the threshold adjacency -> this seems to be the most culprit for a bipartition? 
    thresh_adj_ind = np.arange(len(feature_names))[np.hstack(['thresh_adj' in feat_name for feat_name in feature_names])]
    feature_inds = np.setdiff1d(np.arange(len(feature_names)), thresh_adj_ind) # this step seems a must must. 
    # feature_inds = np.arange(len(feature_names))
    feature_names = feature_names[feature_inds]
    all_feats = all_feats[:,feature_inds]

# =============================================================================
# =============================================================================
# #     Now we can preprocess all the features. 
# =============================================================================
# =============================================================================

    # scale normalize curvature features. 
    all_feats, feature_names = preprocess.scale_normalize_curvature_features(all_feats, 
                                                                              feature_names)
    
    # remove all zero features
    all_feats, feature_names = preprocess.remove_zero_features(all_feats, feature_names)    

    
    # remove high var features
    all_feats, feature_names = preprocess.remove_high_variance_features(all_feats, 
                                                                        feature_names, 
                                                                        variance_threshold_sigma=2)

    # transform ECC features 
    all_feats, feature_names, ECT_rbf_feature_tformer = preprocess.kernel_dim_reduction_ECC_features(all_feats, 
                                                                                                      feature_names, 
                                                                                                      n_dim=100, 
                                                                                                      gamma=None, 
                                                                                                      random_state=1)

# =============================================================================
#       Normalize.
# =============================================================================
    from sklearn.preprocessing import StandardScaler, power_transform
        
    sd_tform = StandardScaler()
    all_feats = power_transform(sd_tform.fit_transform(all_feats))

    # temporal variation selection 
    all_feats, feature_names, coeffs = preprocess.select_time_varying_features(all_feats, 
                                                                                feature_names, 
                                                                                metadict['all_object_TP'], 
                                                                                ridge_alpha=1.)

    print(all_feats.shape)
    
    
    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # #     SAM module analysis ( double check the quantities all check out. )
    # =============================================================================
    # =============================================================================
    # =============================================================================

    # measure the contribution 
    (feature_type, feature_scope), (SAM_expr, Scope_expr), (SAM_contribution, scope_contribution) = SAM.compute_SAM_features_and_scope_contribution(all_feats, 
                                                                                                                                                    feature_names,
                                                                                                                                                    random_state=0)
    print(SAM_contribution)
    print(scope_contribution)
    

    # =============================================================================
    #   Use hcluster to discover modules.          
    # =============================================================================
        
    clustermap, SAM_modules, SAM_modules_featurenames, SAM_modules_feature_indices = SAM.hierarchical_cluster_features_into_SAM_modules(all_feats, 
                                                                                                           feature_names, 
                                                                                                           feature_type=feature_type,
                                                                                                           feature_scope=feature_scope,
                                                                                                           hcluster_heatmap_color='vlag',
                                                                                                           hcluster_method='average', 
                                                                                                           hcluster_metric='euclidean',
                                                                                                           lam_smooth=1e1, 
                                                                                                           p_smooth=0.5, 
                                                                                                           niter_smooth=10,
                                                                                                           min_peak_distance=5,
                                                                                                           debugviz=False)
    
    
    # =============================================================================
    #     Obtain the characteristic signature for each module 
    # =============================================================================
    
    feature_module_expr, feature_module_expr_contrib = SAM.compute_individual_feature_contributions_in_SAM_modules(all_feats, 
                                                                                                                    SAM_modules_feature_indices,
                                                                                                                    random_state=0)
        
    # =============================================================================
    #     grab the top patches that are enriched in each modules. 
    # =============================================================================
    # create a regular. 
    # all_object_patches = metadict['']
    
    
    object_SAM_module_purity_scores, sample_images_modules, sample_object_index_modules = SAM.compute_most_representative_image_patches( feature_module_expr,
                                                                                                                                        all_patches[0], 
                                                                                                                                        n_rows = 3, 
                                                                                                                                        n_cols = 3,
                                                                                                                                        rescale_intensity=False)
    
    # =============================================================================
    #      Now we do the dynamic phenotyping analysis
    # =============================================================================
    
    import umap 
    import seaborn as sns 
    umap_fit = umap.UMAP(n_neighbors=100, 
                          random_state=0,
                          # init=pca_proj, 
                          metric='euclidean') # this seems to look better?
    uu = umap_fit.fit_transform(all_feats) 
    
    
    """
    plot the cropped patches onto the umap coordinates.
    """
    fig, ax = plt.subplots(figsize=(15,15))
    
    plotting.plot_image_patches(positions=uu,
                                patches=metadict['all_object_patches'],
                                ax=ax,
                                subsample=1,
                                zoom=0.35)
    plt.show()
        
    
    
    """
    plot the density map 
    """
    
    all_pts_density, all_pts_density_select = SAM.compute_heatmap_density_image_Embedding(uu, 
                                                                                            all_conditions=None, 
                                                                                            unique_conditions=None,
                                                                                            cmap='coolwarm',
                                                                                            grid_scale_factor=500, 
                                                                                            sigma_factor=0.25, 
                                                                                            contour_sigma_levels=[1,2,3,3.5,4, 4.5, 5],
                                                                                            saveplotsfolder=None)
                  
    """
    select certain statistics and plot -- as they may have disappeared, these need to be based on the prefiltered but still normalized
    """
    area_ind = np.arange(len(feature_names))[np.hstack(['equivalent_diameter' in name for name in feature_names])]
    ecc_ind = np.arange(len(feature_names))[np.hstack(['major_minor_axis_ratio' in name for name in feature_names])]
    flow_ind = np.arange(len(feature_names))[np.hstack(['mean_speed_global' in name for name in feature_names])]
    intensity_ind = np.arange(len(feature_names))[np.hstack(['mean_intensity' in name for name in feature_names])]

    if len(area_ind) > 0 :
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Area '+feature_names[area_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,area_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        # plt.savefig(os.path.join(saveplotsfolder_cells, 
                                  # 'umap_area.svg'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if len(ecc_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Eccentricity '+feature_names[ecc_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,ecc_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        # plt.savefig(os.path.join(saveplotsfolder_cells, 
                              # 'umap_eccentricity.svg'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if len(flow_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Speed '+feature_names[flow_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,flow_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        # plt.savefig(os.path.join(saveplotsfolder_cells, 
                          # 'umap_mean_speed_flow.svg'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if len(intensity_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Intensity '+feature_names[intensity_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,intensity_ind[0]], cmap='coolwarm', vmin=-0.5, vmax=0.5)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        # plt.savefig(os.path.join(saveplotsfolder_cells, 
                          # 'umap_mean_intensity.svg'), dpi=300, bbox_inches='tight')    
        plt.show()
            
# =============================================================================
#     Plot the sam module expr onto the umap. 
# =============================================================================

    for mod_ii in np.arange(feature_module_expr.shape[-1]):    
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title('Module '+str(mod_ii))
        ax.scatter(uu[:,0], 
                    uu[:,1], c=feature_module_expr[:,mod_ii], cmap='coolwarm', vmin=-6, vmax=6)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        # plt.savefig(os.path.join(saveplotsfolder_cells, 
                          # 'umap_module-%s.svg' %(str(mod_ii).zfill(3))), dpi=300, bbox_inches='tight')    
        plt.show()
    
    
# =============================================================================
#     Plot the division!
# =============================================================================

    # all_division
    fig, ax = plt.subplots(figsize=(15,15))
    plt.title('Division times')
    ax.scatter(uu[:,0], 
                uu[:,1], c=all_division, cmap='coolwarm', vmin=-1, vmax=1)
    ax.scatter(uu[all_division==1,0], 
                uu[all_division==1,1], c='r', s=100,zorder=1000)
#    ax.set_aspect(1)
#    ax.set_xlim([-10,10])
    plt.axis('off')
    plt.grid('off')
    # plt.savefig(os.path.join(saveplotsfolder_cells, 
                      # 'umap_divisions.svg' ), dpi=300, bbox_inches='tight')    
    plt.show()
    
    
    
# =============================================================================
#     Do phenotype clustering prototype analysis. 
# =============================================================================
    
    clust_labels, clusterer, best_k, scan_results = SAM.KMeans_cluster_low_dimensional_embedding_with_scan(uu,
                                                                                                             k_range=(2,20),
                                                                                                             ploton=True)
                                                    
    
  
    # clust_labels = clusterer.predict(uu); # does this make more sense with the actual features? 
    clust_labels = clusterer.predict(uu);
    uniq_clust_labels = np.unique(clust_labels)
    clust_labels_colors = sns.color_palette('Set1', len(uniq_clust_labels))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(uu[:,0], 
                uu[:,1], c='lightgrey')
    for lab_ii, lab in enumerate(uniq_clust_labels):
        ax.scatter(uu[clust_labels==lab,0], 
                    uu[clust_labels==lab,1], c=clust_labels_colors[lab_ii], alpha=1)
        
        # get the mean coordinate for each coordinate and label .
        mean_clust_uu = np.mean(uu[clust_labels==lab],axis=0)
        ax.text(mean_clust_uu[0],
                mean_clust_uu[1], 
                str(lab_ii+1),
                va='center',
                ha='center',
                fontsize=24,
                fontname='Arial', zorder=1000)
        
    plt.axis('off')
    plt.grid('off')
    # plt.savefig(os.path.join(saveplotsfolder_cells, 
                              # 'SAM_umap_kmeans_clusters.pdf'), dpi=300, bbox_inches='tight')
    
    # plt.savefig('Comb_umap_2021-07-06_Brittany_batch_correct_powt_kernelECT_8clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    """
    PC-based representative image patches 
    """
    representative_image_dict = SAM.find_representative_images_per_cluster_PCA(all_feats,
                                                                                uu,
                                                                                all_patches_std_size, 
                                                                               all_patches_sizes,
                                                                               clust_labels, 
                                                                               unique_cluster_labels=np.unique(clust_labels), 
                                                                               rev_sign=False,
                                                                               mosaic_size=0.95,
                                                                               percentile_cutoff = 1., # i.e. 100% 
                                                                               n_rows_mosaic = 2, 
                                                                               n_cols_mosaic = 2,
                                                                               pca_random_state=0,
                                                                               debugviz=False)
        
    representative_images_std_size = representative_image_dict['representative_images_std_size']
    representative_images_real_size = representative_image_dict['representative_images_real_size']
    
    
    # vis
    
    for cluster_ii in np.arange(len(representative_images_real_size)):
        
        plt.figure(figsize=(10,10))
        plt.title(cluster_ii)
        plt.imshow(representative_images_real_size[cluster_ii], cmap='gray')
        plt.show()
    
    
    
    """
    SAM module expression per cluster
    """
    unique_clust_labels, cluster_mean_scores, cluster_std_scores = SAM.compute_mean_score_features_cluster(clust_labels, 
                                                                                                           all_feats=feature_module_expr, 
                                                                                                           avg_func=np.nanmean)
    
    
    # plot the expression per cluster. Since the number of phenotype clusters is fairly low, we can use the joint canvas option.
    plotting.barplot_cluster_statistics(cluster_mean_scores,
                                        featnames=['Module %s' %(str(jj).zfill(3)) for jj in np.arange(cluster_mean_scores.shape[1])],
                                        colormap=plt.cm.gray,
                                        style='hbar',
                                        shared_canvas = True,
                                        # figsize=(1,4),
                                        figsize=(10,4),
                                        vmin=-6, 
                                        vmax=6,
                                        nticks=5,
                                        save_dpi=300, 
                                        saveplotsfolder=None)
    
    
    """
    Temporal stacked barplots of cluster 
    """
    uniq_cluster_conditions, proportion_time_condition = SAM.compute_temporal_cluster_proportions(clust_labels, 
                                                                                                    objects_time=metadict['all_object_TP'], 
                                                                                                    time_intervals=np.linspace(0, np.max(metadict['all_object_TP']), 16+1),
                                                                                                    all_conditions=None, 
                                                                                                    unique_conditions=None)

    # plot the stacked barplots as in the paper. 
    clust_labels_colors = sns.color_palette('Set1', len(np.unique(clust_labels)))
    plotting.stacked_barplot_temporal_cluster_proportions(proportion_time_condition, 
                                                         unique_conditions=['U373'], 
                                                         time_intervals = np.linspace(0, np.max(metadict['all_object_TP']), 16+1), 
                                                         clust_labels_colors=clust_labels_colors,
                                                         figsize=(7,5),
                                                         saveplotsfolder=None)
    
    
    """
    Phenotype trajectory
    """
    all_phenotype_trajectories, all_density_contours = SAM.compute_phenotypic_trajectory(uu, 
                                                                                            objects_time=metadict['all_object_TP'], 
                                                                                            time_intervals=np.linspace(0, np.max(metadict['all_object_TP']), 16+1),
                                                                                            all_conditions=None, 
                                                                                            unique_conditions=None,
                                                                                            cmap='coolwarm',
                                                                                            grid_scale_factor=500, 
                                                                                            sigma_factor=0.25, 
                                                                                            thresh_density_sigma = 3,
                                                                                            debugviz=False)
    
    # plot the trajectory atop the points. 
    plt.figure(figsize=(10,10))
    plt.scatter(uu[:,0], 
                uu[:,1], s=20, c='lightgrey')
    plt.plot(all_phenotype_trajectories[0][:,0], 
             all_phenotype_trajectories[0][:,1], 
             'ko-', lw=3, ms=5)
    plt.plot(all_phenotype_trajectories[0][0,0], 
             all_phenotype_trajectories[0][0,1], 
             'ro', ms=15, mec='k', mew=3)
    plt.grid('off')
    plt.axis('off')
    plt.show()
    
    
    
    # =============================================================================
    # =============================================================================
    # #     Test we can do the HMM analysis and single cell trajectory transition analysis
    # =============================================================================
    # =============================================================================
      
    # build the object trajectories.
    object_trajectories_dict = SAM.construct_obj_traj_from_uniq_obj_ids(metadict['all_object_uniq_row_ids'], 
                                                                          separator='_', 
                                                                          wanted_entries=[1,2,3,4], # we need uniq_filename, filename, organoid_no, frame_no
                                                                          d_Frame = 1)
    
    all_obj_trajectories = object_trajectories_dict['all_obj_trajectories']
    
    # check this is correct. 
    # use the phenotype clusters to label the trajectory
    all_object_label_trajectories = SAM.get_labels_for_trajectory( all_obj_trajectories,
                                                                   clust_labels,
                                                                   all_conditions=None, 
                                                                   all_unique_conditions = None)
                  
    """
    HMM learning on the labeled trajectories 
    """
    # only one condition. 
    all_HMM_models = SAM.fit_categoricalHMM_model_to_phenotype_cluster_label_trajectories( all_object_label_trajectories,
                                                                                          clust_labels,
                                                                                          hmm_algorithm = 'map', 
                                                                                          hmm_random_state = 0,
                                                                                          hmm_implementation='scaling')
       
    transition_matrix = all_HMM_models[0][0]
    hmm_model = all_HMM_models[0][1]
    
    
    """
    visualize the transition matrix
    """
    fig, ax = plt.subplots(figsize=(10,10))
    SAM.draw_HMM_transition_graph(transition_matrix, 
                                  ax=ax, 
                                  node_colors=clust_labels_colors, 
                                  node_list=np.arange(len(transition_matrix)), 
                                  edgescale=10, 
                                  edgelabelpos=.5, 
                                  figsize=(15,15),
                                  savefile=None)
    plt.show()
    

# #     """
# #     inject code to plot the markov transitions as equal width arrows. 
# #     """
# #     from sam_analysis import draw_HMM_transition_graph
    
# #     draw_HMM_transition_graph(Q_sort_table, 
# #                               ax=ax, 
# #                               node_colors=clust_labels_colors, 
# #                               node_list=np.arange(len(Q_sort_table)), 
# #                               edgescale=10, 
# #                               edgelabelpos=.5, 
# #                               figsize=(15,15),
# #                               savefile=os.path.join(saveplotsfolder_cells, 
# #                                                     'HMM_Q_transition_ordered_markov_transition_thick_equal-len_arrows.svg'))
    
    
    
# #     import networkx as nx 
    
# #     G = nx.MultiDiGraph()
# #     labels={}
# #     edge_labels={}
    
# #     for i, origin_state in enumerate(states):
# #         for j, destination_state in enumerate(states):
# #             print(i,j)
# #             rate = Q[i][j]
# #             print(rate)
# #             if rate > 0.05:
# #                 G.add_edge(origin_state, destination_state, weight=rate, label="{:.02f}".format(rate))
# #                 edge_labels[(origin_state, destination_state)] = label="{:.02f}".format(rate)
    
# #     plt.figure(figsize=(10,10))
# #     node_size = 200
# #     pos=nx.circular_layout(G)
# #     # # pos = {state:list(state) for state in states}
# #     nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
# #     nx.draw_networkx_labels(G, pos, 
# #                             font_weight=2,
# #                             labels={out_inds[kk]:str((in_inds[kk]+1)) for kk in np.arange(len(out_inds))})
# #     nx.draw_networkx_edge_labels(G, pos, edge_labels)
# #     nx.draw_networkx_nodes(G, pos, label={out_inds[kk]:str((in_inds[kk]+1)) for kk in np.arange(len(out_inds))})
# #     plt.axis('off');
# #     plt.savefig(os.path.join(os.path.join(saveplotsfolder_cells,
# #                              'mc_transition.pdf')), dpi=600, bbox_inches='tight')
# #     plt.show()
    
# #     # from networkx.drawing.nx_pydot import write_dot
# #     # write_dot(G, os.path.join(saveplotsfolder_cells, 'mc_transition.dot'))
    
# #     # from subprocess import check_call
# #     # nfile = 'w.png' 
# #     # check_call(['dot', '-Tpng', 'mc.dot', '-o', nfile])
    
# #     # import matplotlib.image as mpimg
# #     # img = mpimg.imread(nfile)
# #     # plt.axis('off')
# #     # plt.imshow(img)
# #     # plt.show()




# #     for test_patch_id in np.arange(len(all_organoid_trajectories)):
        
# #         # test_patch_id = 10 # 1?    
# #         # test_traj_patch = (all_patches[0][keep_index])[all_organoid_trajectories[test_patch_id]].copy()
# #         test_traj_patch = (all_patches[0])[all_organoid_trajectories[test_patch_id]].copy()
# #         test_traj_clust_labels = clust_labels[all_organoid_trajectories[test_patch_id]].copy()
# #         test_traj_clust_labels = hmm_model.decode(test_traj_clust_labels.reshape(-1,1))[1]
        
# #         #### reencode in original!
# #         test_traj_clust_labels = in_inds[test_traj_clust_labels]
        
# #         # all_label_seqs.append(test_traj_clust_labels)
# #         # # try filtering this. will need to do this over the whole 18 seqs.... 
# #         # hmm_model = CategoricalHMM(n_components=len(np.unique(clust_labels)))
# #         # hmm_model.fit(test_traj_clust_labels.reshape(-1,1))
# #         test_traj_clust_labels_color = np.vstack(clust_labels_colors)[test_traj_clust_labels].copy()
        
        
# #         plt.figure(figsize=(10,10))
# #         plt.imshow(test_traj_clust_labels_color[None,:])
# #         plt.axis('off')
# #         plt.grid('off')
# #         plt.savefig(os.path.join(saveplotsfolder_cells_traj, 
# #                                  'labels_seq_Tra-%s_HMM_filter.svg' %(str(test_patch_id).zfill(3))), dpi=300, bbox_inches='tight')
# #         plt.show()
        
# #         """
# #         Show the mapping in umap space
# #         """
# #         fig, ax = plt.subplots(figsize=(15,15))
# #         # # plt.title('Area')
# #         # ax.scatter(uu[:,0], 
# #         #             uu[:,1], c='k')
        
# #         for lab_ii, lab in enumerate(uniq_clust_labels):
# #             # ax.scatter(uu[clust_labels==lab,0], 
# #                         # uu[clust_labels==lab,1], c=clust_labels_colors[lab_ii], alpha=1)
# #             ax.scatter(uu[clust_labels==lab,0], 
# #                         uu[clust_labels==lab,1], c='lightgrey', alpha=1)
            
# #         traj = uu[all_organoid_trajectories[test_patch_id]]
        
# #         if len(traj)>=2:
# #             traj_colors = np.vstack(sns.color_palette('coolwarm', len(traj)-1))
# #             segs = np.array([traj[:-1], traj[1:]])
            
# #             for seg_ii in np.arange(segs.shape[1]):
# #                 plt.plot(segs[:,seg_ii][:,0],
# #                          segs[:,seg_ii][:,1],
# #                                   lw=5, c=traj_colors[seg_ii])
            
# #         # plt.plot(uu[all_organoid_trajectories[test_patch_id],0], 
# #                   # uu[all_organoid_trajectories[test_patch_id],1], 'ko-', lw=3)
# #             plt.scatter(uu[all_organoid_trajectories[test_patch_id][0],0], 
# #                       uu[all_organoid_trajectories[test_patch_id][0],1], s=500, color='k')
# #         # ax.axis("square")
# #         ax.axis("off")
# #         # ax.set_aspect(1)
# #     #    ax.set_xlim([-10,10])
# #         plt.savefig(os.path.join(saveplotsfolder_cells_traj, 
# #                              'umap_Tra-%s_HMM_filter.svg' %(str(test_patch_id).zfill(3))), dpi=300, bbox_inches='tight')
# #         plt.show()
        
        
# #         """
# #         Show montage with colored labels! 
# #         """
# #         # get the cluster allocation to the cell tracking images. and overlap the coloring!. 
# #         clust_labels_colors = np.vstack(clust_labels_colors)
# #         clust_labels_colors_traj = clust_labels_colors[test_traj_clust_labels]
        
        
# #         # montage.... 
# #         all_patches_sizes = [p.shape for p in test_traj_patch]
# #         mean_I = []
# #         for r_ind in np.arange(len(all_patches_sizes)):
# #             img_panel_r_ind = test_traj_patch[r_ind]
# #             img_panel_r_ind = skexposure.rescale_intensity(img_panel_r_ind)
# #             # mean_I.append(np.mean(img_panel_r_ind))
# #             mean_I.append(np.max(img_panel_r_ind))
# #         all_patches_sizes = np.vstack(all_patches_sizes)
# #         patch_m, patch_n = np.max(all_patches_sizes, axis=0)
        
        
# #         size_panel = int(np.ceil(np.sqrt(len(test_traj_patch))))
# #         n_rows, n_cols = (size_panel,size_panel)
        
# #         # img_panel = np.ones((n_rows*patch_m, 
# #         #                       n_cols*patch_n)) * np.mean(mean_I)
# #         img_panel = np.ones((n_rows*patch_m, 
# #                               n_cols*patch_n)) * np.percentile(mean_I, 75)
        
# #         img_panel_color = np.ones((n_rows*patch_m, 
# #                                     n_cols*patch_n,3))
        
# #         for ii in np.arange(n_rows):
# #             for jj in np.arange(n_cols):
# #                 kk = ii*n_cols + jj
                
# #                 if kk < len(test_traj_patch):
# #                     patch = test_traj_patch[kk]
# #                     patch = skexposure.rescale_intensity(patch)
# #                     # center this.
# #                     mm, nn = patch.shape[:2]
# #                     offset_mm = (patch_m - mm)//2
# #                     offset_nn = (patch_n - nn)//2
                    
# #                     img_panel[ii*patch_m+offset_mm:ii*patch_m+patch.shape[0]+offset_mm, 
# #                               jj*patch_n+offset_nn:jj*patch_n+patch.shape[1]+offset_nn] = patch.copy()
                    
# #                     img_panel_color[ii*patch_m:(ii+1)*patch_m, 
# #                                     jj*patch_n:(jj+1)*patch_n] = clust_labels_colors_traj[kk][None,:]
        
# #         img_panel = np.uint8(255*skexposure.rescale_intensity(img_panel)) #img_panel = np.uint8(255*skexposure.rescale_intensity(img_panel))
# #         img_panel_color = np.uint8(255*img_panel_color)
        
        
# #         # plt.figure(figsize=(10,10))
# #         # plt.imshow(img_panel, cmap='gray')
# #         # plt.axis('off')
# #         # plt.grid('off')
# #         # plt.savefig(os.path.join(saveplotsfolder_cells_traj, 
# #         #                      'img_panel_Tra-%s.svg' %(str(test_patch_id).zfill(3))), dpi=300, bbox_inches='tight')
# #         # plt.show()
        
        
# #         # plt.figure(figsize=(10,10))
# #         # plt.imshow(img_panel_color)
# #         # plt.axis('off')
# #         # plt.grid('off')
# #         # plt.savefig(os.path.join(saveplotsfolder_cells_traj, 
# #         #                      'img_panel_labelcolor_Tra-%s.svg' %(str(test_patch_id).zfill(3))), dpi=300, bbox_inches='tight')
# #         # plt.show()
        
        
# #         plt.figure(figsize=(10,10))
# #         plt.imshow(img_panel, cmap='gray')
# #         plt.imshow(img_panel_color, alpha=0.5)
# #         plt.axis('off')
# #         plt.grid('off')
# #         plt.savefig(os.path.join(saveplotsfolder_cells_traj, 
# #                              'img_panel_labelcolor-overlay_Tra-%s_HMM_filter.svg' %(str(test_patch_id).zfill(3))), dpi=300, bbox_inches='tight')
# #         plt.show()





# #     # get prediction. 


# #         # hmm.GaussianHMM(n_components=3).fit(X, lengths)
# # # GaussianHMM(...
    
    
# #     # montage_traj_color = np.zeros(test_traj_patch.shape+(3,))
# #     # for dd in np.arange(len(clust_labels_colors_traj)):
# #     #     montage_traj_color[dd,...] = clust_labels_colors_traj[dd][None,None,:].copy()
    
# #     # # montage_traj_color = skimage.util.montage(clust_labels_colors_traj[:,None,None,:], multichannel=True)
# #     # montage_traj_color = skimage.util.montage(montage_traj_color, multichannel=True)
    
# #     # plt.figure(figsize=(15,15))
# #     # plt.imshow(montage_traj, cmap='gray')
# #     # plt.imshow(montage_traj_color, alpha=0.25)
# #     # plt.axis('off')
# #     # plt.show()
    
        
        
# #     """
# #     overlay all trajectories on the UMAP 
# #     """

# #     # for test_patch_id in np.arange(len(all_organoid_trajectories)):

# #     fig, ax = plt.subplots(figsize=(15,15))
# #     # # plt.title('Area')
# #     # ax.scatter(uu[:,0], 
# #     #             uu[:,1], c='k')
    
# #     for lab_ii, lab in enumerate(uniq_clust_labels):
# #         ax.scatter(uu[clust_labels==lab,0], 
# #                     uu[clust_labels==lab,1], c=clust_labels_colors[lab_ii], alpha=1)
    
    
# #     all_traj_labels_colors = []
# #     for test_patch_id in np.arange(len(all_organoid_trajectories)):
        
# #         clust_labels_colors_traj = clust_labels_colors[clust_labels[all_organoid_trajectories[test_patch_id]]]; 
# #         all_traj_labels_colors.append(clust_labels_colors_traj)
        
# #         plt.plot(uu[all_organoid_trajectories[test_patch_id],0], 
# #                   uu[all_organoid_trajectories[test_patch_id],1], 'o-', lw=3)
# #         plt.scatter(uu[all_organoid_trajectories[test_patch_id][0],0], 
# #                   uu[all_organoid_trajectories[test_patch_id][0],1], s=500)
# #             # ax.axis("square")
# #     ax.axis("off")
# #     # ax.set_aspect(1)
# # #    ax.set_xlim([-10,10])
# #     plt.savefig(os.path.join(saveplotsfolder_cells, 
# #                          'all_cell_tracks_on_umap.svg'), dpi=300, bbox_inches='tight')
# #     plt.show()
    
    
# #     # compile all_traj_labels_colors and compile this at the exact time!. 
# #     all_traj_labels_colors_array = np.zeros((len(all_traj_labels_colors), np.max([len(tra) for tra in all_traj_labels_colors]), 3))
# #     for ii in np.arange(len(all_traj_labels_colors_array)):
# #         all_traj_labels_colors_array[ii, :len(all_traj_labels_colors[ii])] = all_traj_labels_colors[ii].copy()
    
    
# #     fig, ax = plt.subplots(figsize=(10,10))
# #     ax.imshow(all_traj_labels_colors_array)
# #     ax.set_aspect('auto')
# #     ax.axis('off')
# #     ax.grid('off')
# #     plt.savefig(os.path.join(saveplotsfolder_cells, 
# #                              'UMAP_labelled_all_cell_tracks_unaligned.svg'), dpi=300, bbox_inches='tight')
# #     plt.show()
    
    
# # # =============================================================================
# # # =============================================================================
# # # #     Ok, so we now build this up..... -> putting all the labels at the correct time 
# # # =============================================================================
# # # =============================================================================
# #     all_traj_labels_colors_array_actual_time = np.zeros((len(all_traj_labels_colors), all_organoid_TP.max(), 3))
    
# #     for ii in np.arange(len(all_traj_labels_colors_array)):
# #         times = np.hstack(all_organoid_trajectories_times[ii])-1
# #         all_traj_labels_colors_array_actual_time[ii, times] = all_traj_labels_colors[ii].copy()
    
# #     fig, ax = plt.subplots(figsize=(10,10))
# #     ax.imshow(all_traj_labels_colors_array_actual_time)
# #     ax.set_aspect('auto')
# #     ax.axis('off')
# #     ax.grid('off')
# #     plt.savefig(os.path.join(saveplotsfolder_cells, 
# #                              'UMAP_labelled_all_cell_tracks_actual_time.svg'), dpi=300, bbox_inches='tight')
# #     plt.show()
    
    

# # # =============================================================================
# # # =============================================================================
# # # #     Compute the density over time and join to make a phenotype trajectory!. 
# # # =============================================================================
# # # =============================================================================
    
# #     organoid_condition_classes = ['cell']
    
# #     grid_pts_image = np.zeros((grid_scale_factor+1, grid_scale_factor+1), dtype=np.float32)
# #     # iterate over the different conditions. and days. 
# #     trajectories_condition = []
# #     contour_trajectories_condition = []
    
# #     all_organoid_days = (all_organoid_TP-1) * 1. #/ 24.
# #     day_bins = np.linspace(0, np.max(all_organoid_days), int(2*8+1)) # we can do 8?
    
# #     for cond_ii in np.arange(len(organoid_condition_classes))[:]:
        
# #         traj_cond_time = []
# #         contour_cond_time = []
        
# #         # select_cond = all_organoid_condition_int==cond_ii
# #         select_cond = np.ones(len(uu), dtype=bool) # all 
        
# #         for day_bin_ii in np.arange(len(day_bins)-1)[:]:
    
# #             select_day = np.logical_and(all_organoid_days>=day_bins[day_bin_ii], 
# #                                         all_organoid_days<=day_bins[day_bin_ii+1])
# #             select = np.logical_and(select_cond, select_day)
            
# #             """
# #             some of these were not filmed long enough!.
# #             """
# #             if np.sum(select) == 0: 
# #                 traj_cond_time.append(np.hstack([np.nan, np.nan]))
# #             else:
# #                 xy = uu_tform[select].copy()
# #                 sigma = 1./4*np.mean(pairwise_distances(uu_tform[select])) # how to get a better estimate... 
                
# #                 grid_pts_xy = grid_pts_image.copy().astype(np.float)
# #                 grid_pts_xy[xy[:,0].astype(np.int), xy[:,1].astype(np.int)] += 1. # have to keep adding.
# #                 grid_pts_xy = skfilters.gaussian(grid_pts_xy, sigma=sigma, mode='reflect', preserve_range=True) # depends on the sigma. 
                
# #                 level1mask = grid_pts_xy>=np.mean(grid_pts_xy)+3*np.std(grid_pts_xy) # 3x or 2x? 
# #                 grid_pt_mean = np.mean(np.array(np.where(level1mask>0)).T, axis=0)
# #                 traj_cond_time.append(grid_pt_mean)
                
# #                 # contour_0 = find_contours(im, np.mean(im)+level*np.std(im))
# #                 plt.figure()
# #                 plt.imshow(grid_pts_xy, cmap='coolwarm')
# #                 plt.plot(grid_pt_mean[1], grid_pt_mean[0], 'ro')
# #                 # for level in [2.8, 3, 3.2]:
# #                 for level in [2, 3]:
# #                     contour_0 = find_contours(grid_pts_xy, np.mean(grid_pts_xy)+level*np.std(grid_pts_xy))
# #                     contour_cond_time.append(contour_0)
# #                     for cnt in contour_0:
# #                         # cnt = cnt/float(grid_size[0]) * (max_lim[0][1] - max_lim[0][0]) + max_lim[0][0]
# #                         # ax.plot(cnt[:, 0],
# #                         #         cnt[:,1], color=color, zorder=zorder, 
# #                         #         alpha=1, lw=3)#,
# #                         plt.plot(cnt[:, 1],
# #                                 cnt[:,0], color='k', zorder=100, 
# #                                 alpha=1, lw=3)#,
                
# #                 plt.show()
            
# #         contour_trajectories_condition.append(contour_cond_time)
# #         trajectories_condition.append(traj_cond_time)
        
# #     trajectories_condition = np.array(trajectories_condition)


# #     trajectories_condition = trajectories_condition / grid_scale_factor *  (max_pt - min_pt)  + min_pt  # - pad
# #     # uu_tform = (uu - min_pt) / (max_pt - min_pt) * grid_scale_factor
            
# #     fig, ax = plt.subplots(figsize=(10,10))
# #     ax.scatter(uu[:,0], 
# #                 uu[:,1],c='lightgrey', s=10) # eccentricity
    
# #     for cond_ii in np.arange(len(organoid_condition_classes)):
# #         traj = trajectories_condition[cond_ii]
        
# #         # # color code this by time! especially for the single ...
# #         # traj = uu[all_organoid_trajectories[test_patch_id]]
        
# #         if len(traj)>=2:
# #             traj_colors = np.vstack(sns.color_palette('coolwarm', len(traj)-1))
# #             segs = np.array([traj[:-1], traj[1:]])
            
# #             for seg_ii in np.arange(segs.shape[1]):
# #                 plt.plot(segs[:,seg_ii][:,0],
# #                          segs[:,seg_ii][:,1],
# #                                   lw=5, c=traj_colors[seg_ii])
                
# #         # if cond_ii == 3: 
# #         #     ax.plot(tra[:,0], 
# #         #             tra[:,1], 'o-', lw=5, 
# #         #             color='k', alpha=1, ms=10, zorder=1000) # eccentricity #.5
            
# #         #     ax.plot(tra[0,0], 
# #         #             tra[0,1], 'o',  
# #         #             color='r', alpha=1, ms=10, zorder=1000) # eccentricity #.5
            
# #         #     ax.plot(tra[-1,0], 
# #         #             tra[-1,1], 'o',  
# #         #             color='g', alpha=1, ms=10, zorder=1000) # eccentricity #.5
# #         # else:
# #         #     ax.plot(tra[:,0], 
# #         #                 tra[:,1], 'o-',lw=5,
# #         #                 color='k', alpha=1, ms=10, zorder=1000) # eccentricity #.1
        
# #         # startpoint
# #         ax.plot(traj[0,0], 
# #                 traj[0,1], 'o',  
# #                 color='k', alpha=1, ms=10, zorder=1000) # eccentricity #.5
        
# #         # endpoint 
# #         ax.plot(traj[-1,0], 
# #                 traj[-1,1], 'o',  
# #                 color='r', alpha=1, ms=10, zorder=1000) # eccentricity #.5
# #     ax.axis('off')
# #     ax.grid('off')
# #     # ax.set_aspect(1)
# #     fig.savefig(os.path.join(saveplotsfolder_cells,
# #                              'umap_mean_trajectory_time_colored.pdf'), bbox_inches='tight', dpi=120)
# #     plt.show()



# #     """
# #     Plotting the overall trajectory in a solid color. 
# #     """
# #     fig, ax = plt.subplots(figsize=(10,10))
# #     ax.scatter(uu[:,0], 
# #                 uu[:,1],c='lightgrey', s=10) # eccentricity
    
# #     for cond_ii in np.arange(len(organoid_condition_classes)):
# #         traj = trajectories_condition[cond_ii]
        
# #         # # color code this by time! especially for the single ...
# #         # traj = uu[all_organoid_trajectories[test_patch_id]]
        
# #         # if len(traj)>=2:
# #         #     traj_colors = np.vstack(sns.color_palette('coolwarm', len(traj)-1))
# #         #     segs = np.array([traj[:-1], traj[1:]])
            
# #         #     for seg_ii in np.arange(segs.shape[1]):
# #         #         plt.plot(segs[:,seg_ii][:,0],
# #         #                  segs[:,seg_ii][:,1],
# #         #                           lw=5, c=traj_colors[seg_ii])
# #         ax.plot(traj[:,0], 
# #                 traj[:,1],
# #                 lw=5,
# #                 color='k')
        
# #         # startpoint
# #         ax.plot(traj[0,0], 
# #                 traj[0,1], 'o',  
# #                 color='k', alpha=1, ms=10, zorder=1000) # eccentricity #.5
        
# #         # endpoint 
# #         ax.plot(traj[-1,0], 
# #                 traj[-1,1], 'o',  
# #                 color='r', alpha=1, ms=10, zorder=1000) # eccentricity #.5
# #     ax.axis('off')
# #     ax.grid('off')
# #     # ax.set_aspect(1)
# #     fig.savefig(os.path.join(saveplotsfolder_cells,
# #                              'umap_mean_trajectory_time_blackwhite.pdf'), bbox_inches='tight', dpi=120)
# #     plt.show()





# #     # label trajectories_condition
# #     traj_conditions_umap_label = clusterer.predict(trajectories_condition[0].astype(np.float32))


# # # =============================================================================
# # # =============================================================================
# # # #     Compute now the distribution of the clusters over time .
# # # =============================================================================
# # # =============================================================================

# #     order_hist = np.arange(len(np.unique(clust_labels)))
# #     day_bins_composition = day_bins.copy()
# #     cond_composition_time_condition = []
    
# #     for cond_ii in np.arange(len(organoid_condition_classes))[:]:
        
# #         cond_composition_time = []
# #         # select_cond = all_organoid_condition_int==cond_ii
# #         select_cond = np.ones(len(uu), dtype=bool)
        
# #         for day_bin_ii in np.arange(len(day_bins_composition)-1)[:]:
    
# #             select_day = np.logical_and(all_organoid_days>=day_bins_composition[day_bin_ii], 
# #                                         all_organoid_days<=day_bins_composition[day_bin_ii+1])
# #             select = np.logical_and(select_cond, select_day)
            
# #             if np.sum(select) == 0:
# #                 cond_composition_time.append(np.hstack([np.nan]*len(uniq_clust_labels)))
# #             else:
# #                 # histogram
# #                 lab_select = clust_labels[select].copy()
# #                 hist_lab = np.hstack([np.sum(lab_select==lab) for lab in uniq_clust_labels]) 
# #                 hist_lab = hist_lab/float(np.sum(hist_lab))
                
# #                 cond_composition_time.append(hist_lab)

# #         cond_composition_time = np.array(cond_composition_time)
# #         cond_composition_time_condition.append(cond_composition_time[:, order_hist]) # this is where the order becomes imposed!
# #     cond_composition_time_condition = np.array(cond_composition_time_condition)
    
    
# #     # for jjjj in np.arange(len(cond_composition_time_condition)):
# #     #     fig, ax = plt.subplots(figsize=(30,8))
# #     #     ax.imshow(cond_composition_time_condition[jjjj].T, cmap='RdYlBu_r', vmin=0, vmax=.25)
# #     #     plt.savefig(os.path.join(global_save_analysis_folder,
# #     #                              'Brittany_'+organoid_condition_classes[jjjj]+'_temporal_composition_order-ecc.svg'), bbox_inches='tight')
# #     #     plt.show()
    
# # # =============================================================================
# # #   create the stacked bar plot version. 
# # # =============================================================================

# #     for jjjj in np.arange(len(cond_composition_time_condition))[:]:
        
# #         # cluster_order = order_hist[::-1] # this has already been applied!!!! in this instance.! check again with Xiaoyue!
# #         cluster_order = np.arange(len(order_hist))[::-1]
# #         data = cond_composition_time_condition[jjjj].copy()
# #         data = data[:, cluster_order].copy() # put in the correct order for plotting. 
# #         # create the cumulative totals for bar graph plotting. 
# #         data_cumsums = np.cumsum(data, axis=1)
# # #        ax.bar(labels, men_means, width, yerr=men_std, label='Men')
# # #        ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
# # #               label='Women')

# #         fig, ax = plt.subplots(figsize=(5,5))
# #         # plt.title(le.classes_[jjjj])
# #         # this requires us to build and program the composition up. in order of the histogram 
# #         for ord_ii in np.arange(data.shape[1]):
# #             if ord_ii>0:
# #                 ax.bar(np.arange(cond_composition_time_condition.shape[1]), 
# #                         data[:,ord_ii], 
# #                         bottom = data_cumsums[:,ord_ii-1], 
# #                         # color=clust_labels_colors[order_hist[::-1][ord_ii]],
# #                         color=clust_labels_colors[::-1][ord_ii],
# #                         width=1,
# #                         edgecolor='k')
# #             else:
# #                 ax.bar(np.arange(cond_composition_time_condition.shape[1]), 
# #                         data[:,ord_ii], 
# #                         # color=clust_labels_colors[order_hist[::-1][ord_ii]],
# #                         color=clust_labels_colors[::-1][ord_ii],
# #                         width=1,
# #                         edgecolor='k')
# #         plt.xlim([0-0.5,data.shape[0]-0.5])
# #         plt.ylim([0,1])
# # #        plt.xticks(np.arange(len(data)+1-.5),
# # #                   day_bins_composition
# # #        ax.imshow(cond_composition_time_condition[jjjj].T, cmap='RdYlBu_r', vmin=0, vmax=.25)
# #         plt.savefig(os.path.join(saveplotsfolder_cells,
# #                                   'temporal_composition_stackedbarplots.svg'), bbox_inches='tight',dpi=300)
# #         plt.show()
        

# # # =============================================================================
# # # =============================================================================
# # # =============================================================================
# # # # # Splitting by videos!!!
# # # =============================================================================
# # # =============================================================================
# # # =============================================================================

    


# #     order_hist = np.arange(len(np.unique(clust_labels)))
# #     day_bins_composition = day_bins.copy()
# #     cond_composition_time_condition = []
    
# #     organoid_condition_classes = np.unique(all_expts)
    
# #     for cond_ii in np.arange(len(organoid_condition_classes))[:]:
        
# #         cond_composition_time = []
# #         select_cond = all_expts==organoid_condition_classes[cond_ii]
# #         # select_cond = np.ones(len(uu), dtype=bool)
        
# #         for day_bin_ii in np.arange(len(day_bins_composition)-1)[:]:
    
# #             select_day = np.logical_and(all_organoid_days>=day_bins_composition[day_bin_ii], 
# #                                         all_organoid_days<=day_bins_composition[day_bin_ii+1])
# #             select = np.logical_and(select_cond, select_day)
            
# #             if np.sum(select) == 0:
# #                 cond_composition_time.append(np.hstack([np.nan]*len(uniq_clust_labels)))
# #             else:
# #                 # histogram
# #                 lab_select = clust_labels[select].copy()
# #                 hist_lab = np.hstack([np.sum(lab_select==lab) for lab in uniq_clust_labels]) 
# #                 hist_lab = hist_lab/float(np.sum(hist_lab))
                
# #                 cond_composition_time.append(hist_lab)

# #         cond_composition_time = np.array(cond_composition_time)
# #         cond_composition_time_condition.append(cond_composition_time[:, order_hist]) # this is where the order becomes imposed!
# #     cond_composition_time_condition = np.array(cond_composition_time_condition)
    
    
# #     # for jjjj in np.arange(len(cond_composition_time_condition)):
# #     #     fig, ax = plt.subplots(figsize=(30,8))
# #     #     ax.imshow(cond_composition_time_condition[jjjj].T, cmap='RdYlBu_r', vmin=0, vmax=.25)
# #     #     plt.savefig(os.path.join(global_save_analysis_folder,
# #     #                              'Brittany_'+organoid_condition_classes[jjjj]+'_temporal_composition_order-ecc.svg'), bbox_inches='tight')
# #     #     plt.show()
    
# # # =============================================================================
# # #   create the stacked bar plot version. 
# # # =============================================================================

# #     for jjjj in np.arange(len(cond_composition_time_condition))[:]:
        
# #         # cluster_order = order_hist[::-1] # this has already been applied!!!! in this instance.! check again with Xiaoyue!
# #         cluster_order = np.arange(len(order_hist))[::-1]
# #         data = cond_composition_time_condition[jjjj].copy()
# #         data = data[:, cluster_order].copy() # put in the correct order for plotting. 
# #         # create the cumulative totals for bar graph plotting. 
# #         data_cumsums = np.cumsum(data, axis=1)
# # #        ax.bar(labels, men_means, width, yerr=men_std, label='Men')
# # #        ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
# # #               label='Women')

# #         fig, ax = plt.subplots(figsize=(5,5))
# #         # plt.title(le.classes_[jjjj])
# #         # this requires us to build and program the composition up. in order of the histogram 
# #         for ord_ii in np.arange(data.shape[1]):
# #             if ord_ii>0:
# #                 ax.bar(np.arange(cond_composition_time_condition.shape[1]), 
# #                         data[:,ord_ii], 
# #                         bottom = data_cumsums[:,ord_ii-1], 
# #                         # color=clust_labels_colors[order_hist[::-1][ord_ii]],
# #                         color=clust_labels_colors[::-1][ord_ii],
# #                         width=1,
# #                         edgecolor='k')
# #             else:
# #                 ax.bar(np.arange(cond_composition_time_condition.shape[1]), 
# #                         data[:,ord_ii], 
# #                         # color=clust_labels_colors[order_hist[::-1][ord_ii]],
# #                         color=clust_labels_colors[::-1][ord_ii],
# #                         width=1,
# #                         edgecolor='k')
# #         plt.xlim([0-0.5,data.shape[0]-0.5])
# #         plt.ylim([0,1])
# # #        plt.xticks(np.arange(len(data)+1-.5),
# # #                   day_bins_composition
# # #        ax.imshow(cond_composition_time_condition[jjjj].T, cmap='RdYlBu_r', vmin=0, vmax=.25)
# #         plt.savefig(os.path.join(saveplotsfolder_cells,
# #                                   'temporal_composition_stackedbarplots_expt-%s.svg' %(jjjj)), bbox_inches='tight',dpi=300)
# #         plt.show()


# # # =============================================================================
# # #       also splitting the trajectories by expt. 
# # # =============================================================================

# #     organoid_condition_classes = np.unique(all_expts)
# #     organoid_condition_colors = np.vstack(sns.color_palette('Set2', len(organoid_condition_classes)))
    
# #     grid_pts_image = np.zeros((grid_scale_factor+1, grid_scale_factor+1), dtype=np.float32)
# #     # iterate over the different conditions. and days. 
# #     trajectories_condition = []
# #     contour_trajectories_condition = []
    
# #     all_organoid_days = (all_organoid_TP-1) * 1. #/ 24.
# #     day_bins = np.linspace(0, np.max(all_organoid_days), int(2*8+1)) # we can do 8?
    
# #     for cond_ii in np.arange(len(organoid_condition_classes))[:]:
        
# #         traj_cond_time = []
# #         contour_cond_time = []
        
# #         select_cond = all_expts==organoid_condition_classes[cond_ii]
# #         # select_cond = np.ones(len(uu), dtype=bool) # all 
        
# #         for day_bin_ii in np.arange(len(day_bins)-1)[:]:
    
# #             select_day = np.logical_and(all_organoid_days>=day_bins[day_bin_ii], 
# #                                         all_organoid_days<=day_bins[day_bin_ii+1])
# #             select = np.logical_and(select_cond, select_day)
            
# #             """
# #             some of these were not filmed long enough!.
# #             """
# #             if np.sum(select) == 0: 
# #                 traj_cond_time.append(np.hstack([np.nan, np.nan]))
# #             else:
# #                 xy = uu_tform[select].copy()
# #                 sigma = 1./4*np.mean(pairwise_distances(uu_tform[select])) # how to get a better estimate... 
                
# #                 grid_pts_xy = grid_pts_image.copy().astype(np.float)
# #                 grid_pts_xy[xy[:,0].astype(np.int), xy[:,1].astype(np.int)] += 1. # have to keep adding.
# #                 grid_pts_xy = skfilters.gaussian(grid_pts_xy, sigma=sigma, mode='reflect', preserve_range=True) # depends on the sigma. 
                
# #                 level1mask = grid_pts_xy>=np.mean(grid_pts_xy)+3*np.std(grid_pts_xy) # 3x or 2x? 
# #                 grid_pt_mean = np.mean(np.array(np.where(level1mask>0)).T, axis=0)
# #                 traj_cond_time.append(grid_pt_mean)
                
# #                 # contour_0 = find_contours(im, np.mean(im)+level*np.std(im))
# #                 plt.figure()
# #                 plt.imshow(grid_pts_xy, cmap='coolwarm')
# #                 plt.plot(grid_pt_mean[1], grid_pt_mean[0], 'ro')
# #                 # for level in [2.8, 3, 3.2]:
# #                 for level in [2, 3]:
# #                     contour_0 = find_contours(grid_pts_xy, np.mean(grid_pts_xy)+level*np.std(grid_pts_xy))
# #                     contour_cond_time.append(contour_0)
# #                     for cnt in contour_0:
# #                         # cnt = cnt/float(grid_size[0]) * (max_lim[0][1] - max_lim[0][0]) + max_lim[0][0]
# #                         # ax.plot(cnt[:, 0],
# #                         #         cnt[:,1], color=color, zorder=zorder, 
# #                         #         alpha=1, lw=3)#,
# #                         plt.plot(cnt[:, 1],
# #                                 cnt[:,0], color='k', zorder=100, 
# #                                 alpha=1, lw=3)#,
                
# #                 plt.show()
            
# #         contour_trajectories_condition.append(contour_cond_time)
# #         trajectories_condition.append(traj_cond_time)
        
# #     trajectories_condition = np.array(trajectories_condition)


# #     trajectories_condition = trajectories_condition / grid_scale_factor *  (max_pt - min_pt)  + min_pt  # - pad
# #     # uu_tform = (uu - min_pt) / (max_pt - min_pt) * grid_scale_factor
            
# #     fig, ax = plt.subplots(figsize=(10,10))
# #     ax.scatter(uu[:,0], 
# #                 uu[:,1],c='lightgrey', s=10) # eccentricity
    
# #     for cond_ii in np.arange(len(organoid_condition_classes)):
# #         tra = trajectories_condition[cond_ii]
        
# #         if cond_ii == 3: 
# #             ax.plot(tra[:,0], 
# #                     tra[:,1], 'o-', lw=5, 
# #                     color=organoid_condition_colors[cond_ii], alpha=1, ms=10, zorder=1000) # eccentricity #.5
            
# #             ax.plot(tra[0,0], 
# #                     tra[0,1], 'o',  
# #                     color='r', alpha=1, ms=10, zorder=1000) # eccentricity #.5
            
# #             ax.plot(tra[-1,0], 
# #                     tra[-1,1], 'o',  
# #                     color='g', alpha=1, ms=10, zorder=1000) # eccentricity #.5
# #         else:
# #             ax.plot(tra[:,0], 
# #                         tra[:,1], 'o-',lw=5,
# #                         color=organoid_condition_colors[cond_ii], alpha=1, ms=10, zorder=1000) # eccentricity #.1
            
# #             ax.plot(tra[0,0], 
# #                     tra[0,1], 'o',  
# #                     color='r', alpha=1, ms=10, zorder=1000) # eccentricity #.5
            
# #             ax.plot(tra[-1,0], 
# #                     tra[-1,1], 'o',  
# #                     color='g', alpha=1, ms=10, zorder=1000) # eccentricity #.5
# #     ax.axis('off')
# #     ax.grid('off')
# #     # ax.set_aspect(1)
# #     fig.savefig(os.path.join(saveplotsfolder_cells,
# #                              'umap_mean_trajectory_separate_videos.png'), bbox_inches='tight', dpi=120)
# #     plt.show()


