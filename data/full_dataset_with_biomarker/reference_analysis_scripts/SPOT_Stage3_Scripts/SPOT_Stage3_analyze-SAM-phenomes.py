#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:51:44 2024

@author: s205272

Analysis of the drugs + dye dataset, with hoescht + caspase staining 

"""


if __name__=="__main__":
    
    import os 
    import glob
    import numpy as np 
    import pylab as plt 
    import skimage.io as skio
    import skimage.transform as sktform 
    import skimage.exposure as skexposure
    from tqdm import tqdm 
    import scipy.io as spio
    import seaborn as sns 
    
    
    import SPOT.Utility_Functions.file_io as fio 
    import SPOT.Tracking.track as SPOT_track
    import SPOT.Tracking.optical_flow as SPOT_optical_flow
    import SPOT.Features.features as SPOT_SAM_features    
    import SPOT.Image.image as SPOT_image
    import SPOT.Utility_Functions.file_io as fio
    
    # analysis.     
    import SPOT.Analysis.preprocessing as preprocess
    import SPOT.Analysis.sam_analysis as SAM
    import SPOT.Utility_Functions.plotting as plotting
    
    # umap for analysis.
    import umap 
    

    """
    We use only the registered brightfield images to detect organoids
    """
    # specify the export folder (we will work with this. )
    masterimgfolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision'
    saverootfolder = masterimgfolder
    
    sepcond_norm = False # is batch-based norm. 
    # sepcond_norm = True # is batch-based norm. 
    
    
    if sepcond_norm:
        
    # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye_SPOT_results_UMAPneighbors100'
    # fio.mkdir(saveplotsfolder)
    
    # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/21-SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100'
    # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-27_SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_allcond-norm'
        # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-27_SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_sepcond-norm'
        # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-27-30_SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_sepcond-norm_thresh2.5'
        # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-27-30_SPOT-Drug screen-dye_invert/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_sepcond-norm_thresh2.5'
        saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-30_SPOT-Drug screen-dye_invert/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_sepcond-norm_thresh2.5'
        # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-27_SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors200_sepcond-norm'
    else:
        # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-27-30_SPOT-Drug screen-dye/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_allcond-norm'
        # saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-27-30_SPOT-Drug screen-dye_invert/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_allcond-norm'
        saveplotsfolder = r'/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/Ludwig/SAM_paper_2023/2024_NatCommRevision/26-30_SPOT-Drug screen-dye_invert/Caspase_or_cytox_dye_redo_SPOT_results_UMAPneighbors100_allcond-norm'
    
    fio.mkdir(saveplotsfolder)
    
    
    
    # exptfolders = [r'21-SPOT-Drug screen-dye']
    exptfolders = [r'26-SPOT-Drug screen-dye_invert',
                   # r'27-SPOT-Drug screen-dye_invert',
                   r'30-SPOT-Drug screen-dye_invert']
    # imgfolders = os.listdir(masterimgfolder)
    # imgfolders = np.hstack([os.path.join(masterimgfolder,
    #                                       ff) for ff in imgfolders])

    """
    This should be at the level of individual subfolders. 
    """
    analysisfolders = []
    expts = []
    
    for exptfolder in tqdm(exptfolders[:]):
        
        # saveresultsfolder = os.path.join(saverootfolder, exptfolder, 'Caspase_or_cytox_dye', 'Videos_Resize_Register')
        # saveresultsfolder = os.path.join(saverootfolder, exptfolder, 'Caspase_or_cytox_dye_redo', 'Videos_Resize_Register')
        saveresultsfolder = os.path.join(saverootfolder, exptfolder, 'Caspase or Cytotox Only', 'Videos_Resize_Register')
        
        
        # expt = rootname+'_'+basename
        # expt = exptfolder +'_'+'Caspase_or_cytox_dye'
        # expt = exptfolder +'_'+'Caspase_or_cytox_dye_redo'
        expt = exptfolder +'_'+'Caspase or Cytotox Only' # 26-SPOT-Drug screen-dye_Caspase or Cytotox Only_image_features
        # # find further the folders. 
        # subfolders = os.listdir(infolder)
        
        # for ss in subfolders:
        
        #     saveresultsfolder = os.path.join(infolder, ss)    
        expts.append(expt)
        analysisfolders.append(saveresultsfolder)
       
    print('--------')
    # print(list(analysisfolders))
    for ii in np.arange(len(analysisfolders)):
        print(ii, analysisfolders[ii])
    print('--------')


    """
    load all features [# too small? ]
    """
    all_feats, metadict = fio.load_SPOT_features_files(analysisfolders, 
                                                        expts,
                                                        boundaryfile_suffix='_boundaries_smooth_final.mat',
                                                        boundarykey='boundaries',
                                                        patchfile_suffix='_boundaries_smooth_final_img_patches.mat',
                                                        patches_size_key='patch_size',
                                                        shapefeatsfile_suffix='_shape_features.csv',
                                                        appearfeatsfile_suffix='_image_features.csv', 
                                                        motionfeatsfile_suffix='_motion_features.csv',
                                                        read_chunksize=2000)
                                   
        
    """
    Do the full combine across the 2 datasets.
    """
    all_uniq_org_ids = metadict['all_object_uniq_row_ids']
    all_dataset_boundaries = metadict['all_object_boundaries']
    
    all_patches = list(metadict['all_object_patches'])    
    all_patches = np.array(all_patches, dtype=object) # these are already the standard size image. 
    # std_size=(64,64)
    # all_patches_std_size = np.array([sktform.resize(pp, output_shape=std_size, order=1, preserve_range=True) for pp in all_patches[0]])
    all_patches_sizes = metadict['all_object_patches_sizes']
    
    
    print('=======')
    print('loaded features')
    print('=======')
    
    
    # =============================================================================
    # =============================================================================
    # #     Remove features that shouldn't have been computed 
    # =============================================================================
    # =============================================================================
    
    all_treatments = metadict['all_object_condition'].copy() #### these are all the drug treatments
    all_feats = all_feats.astype(np.float64)
    feature_names = metadict['feature_names']    
    
    all_filenames = [ff.split('_')[-5] for ff in all_uniq_org_ids]
    all_dye_names = [ff.split()[-3] for ff in all_filenames] # this is all the dyenames
    
    
    """
    We added extra information to the appearance features. --- we need to retrieve this out as we will use this for overlay, not as part of SAM features for generating the UMAP etc. 
    """
    # dye_feat_names = ['mean_marker_intensity', 'n_spots', 'mean_marker_spot_intensity']
    dye_feat_names = ['mean_marker_intensity', 
                      'mean_marker_intensity_raw',
                      'n_spots', 
                      'mean_marker_spot_intensity', 
                      'mean_marker_spot_intensity_raw']
    
    dye_feat_names_index = []
    dye_feats = []
    
    for jj in np.arange(len(dye_feat_names)):
        select = np.arange(len(feature_names))[feature_names==dye_feat_names[jj]]
        
        dye_feat_names_index.append(select)
        dye_feats.append(all_feats[:,select])
    
    dye_feat_names_index = np.hstack(dye_feat_names_index)
    dye_feats = np.hstack(dye_feats)
    
    
    all_SAM_index = np.setdiff1d(np.arange(len(feature_names)), dye_feat_names_index)
    
    all_feats = all_feats[:,all_SAM_index].copy()
    feature_names = feature_names[all_SAM_index].copy()
    
    print(all_feats.shape)
    print(feature_names.shape)
    
    print('num nan features: ', (np.sum(np.isnan(all_feats), axis=0)>0).sum())
    
    # =============================================================================
    # # =============================================================================
    # #     Now we can preprocess all the features. 
    # # =============================================================================
    # =============================================================================

    # scale normalize curvature features. 
    all_feats, feature_names = preprocess.scale_normalize_curvature_features(all_feats, 
                                                                              feature_names)
    
    # remove all zero features # this is bad ! 
    all_feats, feature_names = preprocess.remove_zero_features(all_feats, feature_names) 
    
    
    all_feats_bak = all_feats.copy()
    feature_names_bak = feature_names.copy()
    
    # remove features with nan variation. 
    all_feats, feature_names = preprocess.remove_nan_variance_features(all_feats, feature_names) 


    # remove high var features 
    all_feats, feature_names = preprocess.remove_high_variance_features(all_feats, 
                                                                        feature_names, 
                                                                        variance_threshold_sigma=2)


    # transform ECC features (this is different)
    all_feats, feature_names, ECT_rbf_feature_tformer = preprocess.kernel_dim_reduction_ECC_features(all_feats.astype(np.float64), 
                                                                                                      feature_names, 
                                                                                                      n_dim=100, 
                                                                                                      gamma=None, # 1. was closer?  
                                                                                                      random_state=1) # not the same? 
    
    # =============================================================================
    #       Normalize (we would need to save this when applying to the training dataset.)
    # =============================================================================
    
    from sklearn.preprocessing import StandardScaler, power_transform
    
    
    """
    For different conditions, this is done on a per condition basis 
    """

    print(np.unique(all_treatments))
    print('========')
    sd_tform = StandardScaler()
    
    # all_feats = power_transform(sd_tform.fit_transform(all_feats))
    
    # perform normalizion on expt basis
    if sepcond_norm:
        all_feats_copy = np.zeros_like(all_feats)
        # for uniq_expt in np.unique(all_treatments):
        for uniq_expt in np.unique(metadict['all_object_expt']):
            # select = all_treatments == uniq_expt
            select = metadict['all_object_expt'] == uniq_expt
            feats_norm = power_transform(sd_tform.fit_transform(all_feats[select]))
            all_feats_copy[select] = feats_norm.copy()
            # 
        all_feats = all_feats_copy.copy()
    
        del all_feats_copy
    else:
        all_feats = power_transform(sd_tform.fit_transform(all_feats))



    # temporal variation selection (only for time data)
    all_feats, feature_names, coeffs = preprocess.select_time_varying_features(all_feats.astype(np.float64), 
                                                                                feature_names, 
                                                                                metadict['all_object_TP'], 
                                                                                ridge_alpha=1.)
    
    print('=======')
    print('all features ready for analysis')
    print('=======')
    print(all_feats.shape)
    
    
    """
    This is the place to save if we wanted. 
    """
    
    
    
    # =============================================================================
    # =============================================================================
    # =============================================================================
    # # #     SAM module analysis ( double check the quantities all check out. ), 700 approx features is too much ?
    # =============================================================================
    # =============================================================================
    # =============================================================================

    # measure the contribution 
    (feature_type, feature_scope), (SAM_expr, Scope_expr), (SAM_contribution, scope_contribution) = SAM.compute_SAM_features_and_scope_contribution(all_feats, 
                                                                                                                                                    feature_names,
                                                                                                                                                    random_state=0)
    print(SAM_contribution)
    print(scope_contribution)
    
    
    # # =============================================================================
    # #   Use hcluster to discover modules. (this takes a long time? ) --- shouldn't be this long.          
    # # =============================================================================
        
    saveclusterimg = os.path.join(saveplotsfolder,
                                  'SAM_cluster_heatmap.svg')
    
    clustermap, SAM_modules, SAM_modules_colorbar, SAM_modules_featurenames, SAM_modules_feature_indices = SAM.hierarchical_cluster_features_into_SAM_modules(all_feats, 
                                                                                                                                                            feature_names, 
                                                                                                                                                            feature_type=feature_type,
                                                                                                                                                            feature_scope=feature_scope,
                                                                                                                                                            hcluster_heatmap_color='vlag',
                                                                                                                                                            hcluster_method='average', 
                                                                                                                                                            hcluster_metric='euclidean',
                                                                                                                                                            max_clust_len=None,
                                                                                                                                                            buffer = 30,
                                                                                                                                                            # lam_smooth=1e1, 
                                                                                                                                                            lam_smooth=1e0, 
                                                                                                                                                            p_smooth=0.5, 
                                                                                                                                                            niter_smooth=10,
                                                                                                                                                            min_peak_distance=5,
                                                                                                                                                            debugviz=True,
                                                                                                                                                            savefile=saveclusterimg) # set a path to save!. 

    plt.figure(figsize=(15,15))
    plt.imshow(SAM_modules_colorbar[None,...])
    plt.savefig(os.path.join(saveplotsfolder,
                                  'SAM_cluster_heatmap_colorbar.svg'), dpi=300, bbox_inches='tight')
    plt.show()



    # =============================================================================
    #     Obtain the characteristic signature for each module 
    # =============================================================================
    
    feature_module_expr, feature_module_expr_contrib = SAM.compute_individual_feature_contributions_in_SAM_modules(all_feats, 
                                                                                                                    SAM_modules_feature_indices,
                                                                                                                    random_state=0)

    # =============================================================================
    #     We need to generate barplots of the contribution. of each feature.     
    # =============================================================================
    
    save_module_features_folder = os.path.join(saveplotsfolder,
                                            'SAM_modules_features_contrib')
    fio.mkdir(save_module_features_folder)
    
    
    for ii in np.arange(len(SAM_modules_featurenames)):
        
        sort_order =  np.argsort(np.abs(feature_module_expr_contrib[ii]))[::-1]
        
        
        plt.figure(figsize=(1.*len(sort_order),10))
        plt.bar(np.arange(len(feature_module_expr_contrib[ii])), 
                feature_module_expr_contrib[ii][sort_order],
                width=0.5)
        plt.xticks(np.arange(len(feature_module_expr_contrib[ii])),
                    SAM_modules_featurenames[ii][sort_order],
                    fontname='Liberation Sans',
                    fontsize=14,
                    rotation=90)
        plt.savefig(os.path.join(save_module_features_folder,
                                  'Module_'+str(ii).zfill(6)+'.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
    
    
    # =============================================================================
    #     grab the top patches that are enriched in each modules. 
    # =============================================================================
    # create a regular. 
    # all_object_patches = metadict['']
    
    
    object_SAM_module_purity_scores, sample_images_modules, sample_object_index_modules = SAM.compute_most_representative_image_patches( feature_module_expr,
                                                                                                                                        all_patches, 
                                                                                                                                        n_rows = 3, 
                                                                                                                                        n_cols = 3,
                                                                                                                                        rescale_intensity=False)
        
    # save the sample images 
    save_SAM_patches_folder = os.path.join(saveplotsfolder,
                                            'SAM_modules_images')
    fio.mkdir(save_SAM_patches_folder)
    
    for ii in np.arange(len(sample_images_modules)):
        # save the tifs 
        skio.imsave(os.path.join(save_SAM_patches_folder, 
                                  'SAM_module_'+str(ii).zfill(3)+'.tif'), 
                    np.uint8(sample_images_modules[ii]))
        
    
    
    # =============================================================================
    #      Now we do the phenotyping analysis ( static, slow if setting random_state)
    # ============================================================================= 
    
    import time 
    
    t1 = time.time()
    umap_fit = umap.UMAP( n_neighbors=100, # can increase this 
                            # n_neighbors = 50, 
                           # n_neighbors = 15, 
                          random_state=0,
                          metric='euclidean') # this seems to look better?
    uu = umap_fit.fit_transform(all_feats.astype(np.float64)) 
    t2 = time.time()
    
    print('done umap, ', t2-t1)
    
    
    """
    plot the cropped patches onto the umap coordinates.
    """
    fig, ax = plt.subplots(figsize=(15,15))
    
    plotting.plot_image_patches(positions=uu,
                                patches=all_patches.astype(np.uint8),
                                ax=ax,
                                subsample=10, # sample more. 
                                zoom=0.20)
    plt.axis('off')
    plt.grid('off')
    # plt.square()
    plt.savefig(os.path.join(saveplotsfolder, 
                              'image-patches_UMAP.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    
    
    # """
    # Overlay the cell death markers and visualize on the UMAP  
    # """
    # dye_feat_names = ['mean_marker_intensity', 'n_spots', 'mean_marker_spot_intensity']
    # dye_feat_names_index = []
    # dye_feats = []
    
    # ok.... 
    dye_vmin_vmax = [[-2,2], 
                     [0,255],
                     [0,3], # this is the specification for the  number of spots - so it should be ok. 
                     [-0.2,0.2],
                     [0,1]]
    dye_cmap = ['coolwarm',
                'coolwarm',
                'magma', 
                'coolwarm',
                'coolwarm']
    
    for dye_ii in np.arange(len(dye_feat_names)):
        fig, ax = plt.subplots(figsize=(15,15))
        plt.title(dye_feat_names[dye_ii])
        ax.plot(uu[:,0], 
                uu[:,1], '.', color='lightgrey')
        if dye_ii !=1:
            ax.scatter(uu[:,0],
                       uu[:,1], 
                       cmap=dye_cmap[dye_ii],
                       c = dye_feats[:,dye_ii],
                       vmin=dye_vmin_vmax[dye_ii][0],
                       vmax=dye_vmin_vmax[dye_ii][1],
                       s=5, zorder=100)
        else:
            ax.scatter(uu[:,0],
                       uu[:,1], 
                       cmap=dye_cmap[dye_ii],
                       c = np.log(dye_feats[:,dye_ii]+1.),
                       vmin=dye_vmin_vmax[dye_ii][0],
                       vmax=dye_vmin_vmax[dye_ii][1],
                       s=5, zorder=100)
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder, 
                                  'UMAP_caspase-cytox_joint_%s.png' %(dye_feat_names[dye_ii])), 
                    dpi=600, bbox_inches='tight')
        plt.show()


    # separately show which has cytox which has caspase staining
    uniq_dye_names = np.unique(all_dye_names)
    
    for dye_ii in np.arange(len(uniq_dye_names)):
        fig, ax = plt.subplots(figsize=(15,15))
        ax.plot(uu[:,0], 
                uu[:,1], '.', color='lightgrey')
        
        ax.scatter(uu[np.hstack(all_dye_names)==uniq_dye_names[dye_ii],0],
                   uu[np.hstack(all_dye_names)==uniq_dye_names[dye_ii],1], 
                   color='navy',
                   s=5, zorder=100)
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder, 
                                  'UMAP_caspase-cytox_bool_%s.png' %(uniq_dye_names[dye_ii])), 
                    dpi=600, bbox_inches='tight')
        plt.show()



    # plot the marker signature independently gated by dyename
    for marker_ii in np.arange(len(uniq_dye_names)):
        
        select_marker = np.hstack(all_dye_names) == uniq_dye_names[marker_ii]
        
        for dye_ii in np.arange(len(dye_feat_names)):
            fig, ax = plt.subplots(figsize=(15,15))
            ax.plot(uu[:,0], 
                    uu[:,1], '.', color='lightgrey')
            if dye_ii !=1:
                ax.scatter(uu[select_marker,0],
                           uu[select_marker,1], 
                           cmap=dye_cmap[dye_ii],
                           c = dye_feats[select_marker,dye_ii],
                           vmin=dye_vmin_vmax[dye_ii][0],
                           vmax=dye_vmin_vmax[dye_ii][1],
                           s=5, zorder=100)
            else:
                ax.scatter(uu[select_marker,0],
                           uu[select_marker,1], 
                           cmap=dye_cmap[dye_ii],
                           c = np.log(dye_feats[select_marker,dye_ii]+1.),
                           vmin=dye_vmin_vmax[dye_ii][0],
                           vmax=dye_vmin_vmax[dye_ii][1],
                           s=5, zorder=100)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(os.path.join(saveplotsfolder, 
                                      'UMAP_caspase-cytox_joint_%s-marker_%s.png' %(dye_feat_names[dye_ii], uniq_dye_names[marker_ii])), 
                        dpi=600, bbox_inches='tight')
            plt.show()

    
    """
    plot the density map 
    """
    saveplotsdensity = os.path.join(saveplotsfolder, 
                                    'Drug-Conditions')
    fio.mkdir(saveplotsdensity)
    all_pts_density, all_pts_density_select = SAM.compute_heatmap_density_image_Embedding(uu, 
                                                                                            all_conditions=all_treatments, 
                                                                                            unique_conditions=np.unique(all_treatments),
                                                                                            cmap='coolwarm',
                                                                                            grid_scale_factor=500, 
                                                                                            sigma_factor=0.25, 
                                                                                            contour_sigma_levels=[1,2,3,3.5,4, 4.5, 5],
                                                                                            saveplotsfolder=saveplotsdensity)
       
    
    """
    select certain statistics and plot -- as they may have disappeared, these need to be based on the prefiltered but still normalized
    """
    area_ind = np.arange(len(feature_names))[np.hstack(['equivalent_diameter' in name for name in feature_names])]
    ecc_ind = np.arange(len(feature_names))[np.hstack(['major_minor_axis_ratio' in name for name in feature_names])]
    flow_ind = np.arange(len(feature_names))[np.hstack(['mean_speed_global' in name for name in feature_names])]
    intensity_ind = np.arange(len(feature_names))[np.hstack(['mean_intensity' in name for name in feature_names])]

    if len(area_ind) > 0 :
        fig, ax = plt.subplots(figsize=(15,15))
        # plt.title('Area '+feature_names[area_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,area_ind[0]], cmap='coolwarm', vmin=-3, vmax=3)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder, 
                                   'umap_area.png'), dpi=600, bbox_inches='tight')
        plt.show()
    
    if len(ecc_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        # plt.title('Eccentricity '+feature_names[ecc_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,ecc_ind[0]], cmap='coolwarm', vmin=-3, vmax=3)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder, 
                               'umap_eccentricity.png'), dpi=600, bbox_inches='tight')
        plt.show()
    
    if len(flow_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        # plt.title('Speed '+feature_names[flow_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,flow_ind[0]], cmap='coolwarm', vmin=-3, vmax=3)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder, 
                           'umap_mean_speed_flow.png'), dpi=600, bbox_inches='tight')
        plt.show()
    
    if len(intensity_ind)>0:
        fig, ax = plt.subplots(figsize=(15,15))
        # plt.title('Intensity '+feature_names[intensity_ind[0]])
        ax.scatter(uu[:,0], 
                    uu[:,1], c=all_feats[:,intensity_ind[0]], cmap='coolwarm', vmin=-3, vmax=3)
    #    ax.set_aspect(1)
    #    ax.set_xlim([-10,10])
        plt.axis('off')
        plt.grid('off')
        plt.savefig(os.path.join(saveplotsfolder, 
                           'umap_mean_intensity.png'), dpi=600, bbox_inches='tight')    
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
        plt.savefig(os.path.join(saveplotsfolder, 
                            'umap_module-%s.png' %(str(mod_ii).zfill(3))), dpi=600, bbox_inches='tight')    
        plt.show()
    
# =============================================================================
#     Do phenotype clustering prototype analysis. 
# =============================================================================
    
    clust_labels, clusterer, best_k, scan_results = SAM.KMeans_cluster_low_dimensional_embedding_with_scan(uu,
                                                                                                              k_range=(2,20),
                                                                                                              ploton=True)
                                                    
    
  
    # clust_labels = clusterer.predict(uu); # does this make more sense with the actual features? 
    clust_labels = clusterer.predict(uu);
    
    
# =============================================================================
# =============================================================================
# #     reorder the clusters by eccentricity. 
# =============================================================================
# =============================================================================
    clust_labels, cluster_labels_mapping = SAM.reorder_and_relabel_clusters(clust_labels, 
                                                                            score = all_feats[:,ecc_ind[0]], 
                                                                            order='ascending')
    
    
    uniq_clust_labels = np.unique(clust_labels)
    # clust_labels_colors = sns.color_palette('Set1', len(uniq_clust_labels))
    clust_labels_colors = sns.color_palette('rainbow', len(uniq_clust_labels))


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
    plt.savefig(os.path.join(saveplotsfolder, 
                                'SAM_umap_kmeans_clusters.pdf'), dpi=300, bbox_inches='tight')
    
    # plt.savefig('Comb_umap_2021-07-06_Brittany_batch_correct_powt_kernelECT_8clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(uu[:,0], 
                uu[:,1], c='lightgrey')
    for lab_ii, lab in enumerate(uniq_clust_labels):
        ax.scatter(uu[clust_labels==lab,0], 
                    uu[clust_labels==lab,1], c=clust_labels_colors[lab_ii], alpha=1)
        
    plt.axis('off')
    plt.grid('off')
    plt.savefig(os.path.join(saveplotsfolder, 
                                'SAM_umap_kmeans_clusters_notext.png'), dpi=600, bbox_inches='tight')
    
    # plt.savefig('Comb_umap_2021-07-06_Brittany_batch_correct_powt_kernelECT_8clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    """
    PC-based representative image patches 
    """
    representative_image_dict = SAM.find_representative_images_per_cluster_PCA(all_feats,
                                                                                uu,
                                                                                all_patches, 
                                                                                all_patches_sizes,
                                                                                clust_labels, 
                                                                                unique_cluster_labels=np.unique(clust_labels), 
                                                                                rev_sign=False,
                                                                                mosaic_size=0.98,
                                                                                percentile_cutoff = 0.75, # i.e. 100% 
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
        plt.savefig(os.path.join(saveplotsfolder, 
                                    'SAM_umap_kmeans_clusters_%s.pdf' %(str(cluster_ii).zfill(3))), dpi=300, bbox_inches='tight')
        plt.show()
    
    
    # save out using std size. 
    for cluster_ii in np.arange(len(representative_images_std_size)):
        
        plt.figure(figsize=(10,10))
        plt.title(cluster_ii)
        plt.imshow(representative_images_std_size[cluster_ii], cmap='gray')
        plt.savefig(os.path.join(saveplotsfolder, 
                                    'SAM_umap_kmeans_clusters_%s_std-size.pdf' %(str(cluster_ii).zfill(3))), dpi=300, bbox_inches='tight')
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
                                        vmin=-8, 
                                        vmax=8,
                                        nticks=5,
                                        save_dpi=300, 
                                        saveplotsfolder=saveplotsfolder)
    

    # """
    # An evaluation of the mean marker intensity and spot distribution in each cluster and we make plots, to aid interpretation. 
    # """
    
    """
    Assess mean over treatments. 
    """
    import scipy.stats as spstats
    
    uniq_treatments = np.unique(all_treatments)
    
    # 1. compute the proportions of clusteres in each cell-line
    cluster_marker_mean = []
    cluster_marker_std = []
    
    # iterate over unique treatements
    for cc_ii, cc in enumerate(uniq_treatments): # not treatment (which is useless.)
        
        print(cc_ii, cc)
        print('====')
        data = []
        std = []
        select_treat = all_treatments == cc
        
        for dd_ii, dd in enumerate(uniq_dye_names):
            select_marker = np.hstack(all_dye_names) == dd
        
            data_dye = []
            std_dye = []
            
            for metric_ii, marker in enumerate(dye_feat_names):
                
                data_marker = np.nanmean(dye_feats[np.logical_and(select_treat, select_marker), metric_ii])
                std_marker = 2*spstats.sem(dye_feats[np.logical_and(select_treat, select_marker), metric_ii],
                                         nan_policy='omit')
                
                data_dye.append(data_marker)
                std_dye.append(std_marker)
                
            data.append(data_dye)
            std.append(std_dye)   
            
        cluster_marker_mean.append(data)
        cluster_marker_std.append(std)
        
    cluster_marker_mean = np.array(cluster_marker_mean)
    cluster_marker_std = np.array(cluster_marker_std)
        
    
    """
    3 separate barcharts. over the measurements. 
        x = conditions
        xsplit = marker. 
    """
    for jjj in np.arange(cluster_marker_mean.shape[-1]):
        marker_name = dye_feat_names[jjj]
        
        plt.figure(figsize=(5,5))
        plt.bar(np.arange(len(cluster_marker_mean))-0.125, 
                    cluster_marker_mean[:,0,jjj],
                    width=0.125,
                    ec='k', 
                    color='lightgrey')
        plt.errorbar(np.arange(len(cluster_marker_mean))-0.125, 
                    cluster_marker_mean[:,0,jjj], 
                    yerr=cluster_marker_std[:,0,jjj],
                    capthick=1, color='k',fmt='none',capsize=5)
        
        plt.bar(np.arange(len(cluster_marker_mean))+0.125, 
                    cluster_marker_mean[:,1,jjj],
                    width=0.125,
                    ec='k', 
                    color='darkgrey')
        
        plt.errorbar(np.arange(len(cluster_marker_mean))+0.125, 
                    cluster_marker_mean[:,1,jjj], 
                    yerr=cluster_marker_std[:,1,jjj],
                    capthick=1, color='k',fmt='none',capsize=5)
        
        plt.tick_params(length=10, right=True)
        plt.savefig(os.path.join(saveplotsfolder, 
                                 'mean_and_sem_expr_per_treatment_marker-'+dye_feat_names[jjj]+'.svg'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.show()
        
        
    order_treatments = np.hstack([3,2,0,1,4])
    
    for jjj in np.arange(cluster_marker_mean.shape[-1]):
        marker_name = dye_feat_names[jjj]
        
        plt.figure(figsize=(5,5))
        plt.bar(np.arange(len(cluster_marker_mean))-0.125, 
                    cluster_marker_mean[order_treatments,0,jjj],
                    width=0.125,
                    ec='k', 
                    color='lightgrey')
        plt.errorbar(np.arange(len(cluster_marker_mean))-0.125, 
                    cluster_marker_mean[order_treatments,0,jjj], 
                    yerr=cluster_marker_std[order_treatments,0,jjj],
                    capthick=1, color='k',fmt='none',capsize=5)
        
        plt.bar(np.arange(len(cluster_marker_mean))+0.125, 
                    cluster_marker_mean[order_treatments,1,jjj],
                    width=0.125,
                    ec='k', 
                    color='darkgrey')
        
        plt.errorbar(np.arange(len(cluster_marker_mean))+0.125, 
                    cluster_marker_mean[order_treatments,1,jjj], 
                    yerr=cluster_marker_std[order_treatments,1,jjj],
                    capthick=1, color='k',fmt='none',capsize=5)
        
        plt.tick_params(length=10, right=True)
        plt.xticks(np.arange(len(cluster_marker_mean)), 
                   uniq_treatments[order_treatments], rotation=90)
        plt.savefig(os.path.join(saveplotsfolder, 
                                 'mean_and_sem_expr_per_treatment_marker-'+dye_feat_names[jjj]+'_treatment_ordered.svg'), 
                    dpi=300, 
                    bbox_inches='tight')
        plt.show()
    
        
    
    # save a colorbar
    colors = sns.color_palette('magma', 256);
    colors = np.vstack(colors)
    fig, ax = plt.subplots(figsize=(10,10))
    # sns.palplot(sns.color_palette('magma', 256))
    ax.imshow(colors[None,:])
    plt.savefig(os.path.join(saveplotsfolder,
                             'magma_colorbar.svg'), 
                dpi=600, 
                bbox_inches='tight')
    plt.show()
    
        
# =============================================================================
# =============================================================================
# #     over clusters.
# =============================================================================
# =============================================================================
    import scipy.stats as spstats
    uniq_clusters = np.unique(clust_labels)
    
    # 1. compute the proportions of clusteres in each cell-line
    cluster_marker_mean = []
    cluster_marker_std = []
    
    # iterate over unique treatements
    for cc_ii, cc in enumerate(uniq_clusters): # not treatment (which is useless.)
        
        data = []
        std = []
        select_treat = clust_labels == cc
        
        for dd_ii, dd in enumerate(uniq_dye_names):
            select_marker = np.hstack(all_dye_names) == dd
        
            data_dye = []
            std_dye = []
            
            for metric_ii, marker in enumerate(dye_feat_names):
                
                data_marker = np.nanmean(dye_feats[np.logical_and(select_treat, select_marker), metric_ii])
                std_marker = 2*spstats.sem(dye_feats[np.logical_and(select_treat, select_marker), metric_ii],
                                         nan_policy='omit')
                
                data_dye.append(data_marker)
                std_dye.append(std_marker)
                
            data.append(data_dye)
            std.append(std_dye)   
            
        cluster_marker_mean.append(data)
        cluster_marker_std.append(std)
        
    cluster_marker_mean = np.array(cluster_marker_mean)
    cluster_marker_std = np.array(cluster_marker_std)
        
    
    """
    3 separate barcharts. over the measurements. 
        x = conditions
        xsplit = marker. 
    """
    for jjj in np.arange(cluster_marker_mean.shape[-1]):
        marker_name = dye_feat_names[jjj]
        
        plt.figure(figsize=(5,5))
        plt.bar(np.arange(len(cluster_marker_mean))-0.125, 
                    cluster_marker_mean[:,0,jjj],
                    width=0.125,
                    ec='k', 
                    color='lightgrey')
        plt.errorbar(np.arange(len(cluster_marker_mean))-0.125, 
                    cluster_marker_mean[:,0,jjj], 
                    yerr=cluster_marker_std[:,0,jjj],
                    capthick=1, color='k',fmt='none',capsize=5)
        
        plt.bar(np.arange(len(cluster_marker_mean))+0.125, 
                    cluster_marker_mean[:,1,jjj],
                    width=0.125,
                    ec='k', 
                    color='darkgrey')
        
        plt.errorbar(np.arange(len(cluster_marker_mean))+0.125, 
                    cluster_marker_mean[:,1,jjj], 
                    yerr=cluster_marker_std[:,1,jjj],
                    capthick=1, color='k',fmt='none',capsize=5)
        plt.tick_params(length=10, right=True)
        
        plt.savefig(os.path.join(saveplotsfolder, 
                                 'mean_and_sem_expr_per_cluster_marker-'+dye_feat_names[jjj]+'.svg'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    
    
# =============================================================================
# =============================================================================
# #     Temporal transition analysis. 
# =============================================================================
# =============================================================================
    
    """
    Temporal stacked barplots of cluster 
    """
    uniq_cluster_conditions, proportion_time_condition = SAM.compute_temporal_cluster_proportions(clust_labels, 
                                                                                                    objects_time=metadict['all_object_TP']*np.unique(metadict['all_object_Ts'])[0], 
                                                                                                    time_intervals=np.linspace(0, np.max(metadict['all_object_TP']), 16+1)*np.unique(metadict['all_object_Ts'])[0],
                                                                                                    all_conditions=all_treatments, 
                                                                                                    unique_conditions=np.unique(all_treatments))

    # plot the stacked barplots as in the paper. 
    # clust_labels_colors = sns.color_palette('Set1', len(np.unique(clust_labels)))
    clust_labels_colors = sns.color_palette('rainbow', len(np.unique(clust_labels)))
    
    plotting.stacked_barplot_temporal_cluster_proportions(proportion_time_condition, 
                                                         unique_conditions=np.unique(all_treatments), 
                                                         time_intervals = np.linspace(0, np.max(metadict['all_object_TP']), 16+1)*np.unique(metadict['all_object_Ts'])[0], 
                                                         clust_labels_colors=clust_labels_colors,
                                                         figsize=(7,5),
                                                         saveplotsfolder=saveplotsfolder)


    """
    Phenotype trajectory
    """
    all_phenotype_trajectories, all_density_contours = SAM.compute_phenotypic_trajectory(uu, 
                                                                            objects_time=metadict['all_object_TP'], 
                                                                            time_intervals=np.linspace(0, np.max(metadict['all_object_TP']), 16+1),
                                                                            all_conditions=all_treatments, 
                                                                            unique_conditions=np.unique(all_treatments),
                                                                            cmap='coolwarm',
                                                                            grid_scale_factor=500, 
                                                                            sigma_factor=0.25, 
                                                                            # thresh_density_sigma = 3, # use 2. 
                                                                            # thresh_density_sigma = 2.5,# this works
                                                                            thresh_density_sigma = 2., 
                                                                            debugviz=False)
    
    """
    Iterate and plot the phenotype trajectory for all conditions. 
    """
    
    uniq_treatments = np.unique(all_treatments)
    
    for jj in np.arange(len(uniq_treatments)):
    
        # plot the trajectory atop the points. 
        plt.figure(figsize=(10,10))
        plt.title(uniq_treatments[jj])
        plt.scatter(uu[:,0], 
                    uu[:,1], s=20, c='lightgrey')
        plt.plot(all_phenotype_trajectories[jj][:,0], 
                 all_phenotype_trajectories[jj][:,1], 
                 'ko-', lw=3, ms=5)
        plt.plot(all_phenotype_trajectories[jj][0,0], 
                 all_phenotype_trajectories[jj][0,1], 
                 'ro', ms=15, mec='k', mew=3)
        plt.grid('off')
        plt.axis('off')
        plt.savefig(os.path.join(saveplotsfolder,
                                 'population_phenotype_trajectory_%s.png' %(uniq_treatments[jj])), dpi=600, bbox_inches='tight')
        plt.show()
    
    
    """
    hclustering of the trajectories to reveal similarity. 
    """
    # missing - add dtaidistance as dependency.
    trajectory_affinity_matrix= SAM.construct_affinity_matrix_phenotypic_trajectory(all_phenotype_trajectories,
                                                                                    affinity='dtw',
                                                                                    use_std_denom=True)
    
    
    trajectory_cluster_obj = SAM.hcluster_phenotypic_trajectory_affinity_matrix(trajectory_affinity_matrix,
                                                                               uniq_conditions=uniq_treatments,
                                                                               linkage='average', 
                                                                               metric='euclidean',
                                                                               cmap='vlag', 
                                                                               figsize=(10,10),
                                                                               save_dpi=300,
                                                                               savefolder=saveplotsfolder)
    
    
    trajectory_cluster_ward_obj = SAM.hcluster_phenotypic_trajectory_affinity_matrix(trajectory_affinity_matrix,
                                                                               uniq_conditions=uniq_treatments,
                                                                               linkage='ward', 
                                                                               metric='euclidean',
                                                                               cmap='vlag', 
                                                                               figsize=(10,10),
                                                                               save_dpi=300,
                                                                               savefolder=saveplotsfolder)
    
    trajectory_cluster_complete_obj = SAM.hcluster_phenotypic_trajectory_affinity_matrix(trajectory_affinity_matrix,
                                                                               uniq_conditions=uniq_treatments,
                                                                               linkage='complete', 
                                                                               metric='euclidean',
                                                                               cmap='vlag', 
                                                                               figsize=(10,10),
                                                                               save_dpi=300,
                                                                               savefolder=saveplotsfolder)
    
    """
    HMM analysis
    """
    # =============================================================================
    # =============================================================================
    # #     Test we can do the HMM analysis and single cell trajectory transition analysis
    # =============================================================================
    # =============================================================================
      
    # reparse the unique object_ids so we can get the necessary information. 
    all_uniq_org_ids_reparse = []
    for jj in np.arange(len(all_uniq_org_ids)):
        org_id = all_uniq_org_ids[jj].split('_')
        # expt_name = '-'.join(org_id[:6])
        # fname = '-'.join(org_id[6:9])
        # org_no = org_id[9]
        # frame_no = org_id[10]
        expt_name = '-'.join(org_id[:3])
        fname = '-'.join(org_id[3:6])
        org_no = org_id[6]
        frame_no = org_id[7]
        
        new_org_id = expt_name+'_'+fname+'_'+org_no+'_'+frame_no
        
        all_uniq_org_ids_reparse.append(new_org_id)
    all_uniq_org_ids_reparse = np.hstack(all_uniq_org_ids_reparse)
    
    assert(len(all_uniq_org_ids_reparse) == len(np.unique(all_uniq_org_ids_reparse))) # the information should still be retained. 
    
    # build the object trajectories.
    object_trajectories_dict = SAM.construct_obj_traj_from_uniq_obj_ids(all_uniq_org_ids_reparse, # we need uniq_filename, filename, organoid_no, frame_no
                                                                          separator='_', 
                                                                          wanted_entries=[0,1,2], # we need uniq_filename, filename, organoid_no
                                                                          d_Frame = 1)
    
    all_obj_trajectories = object_trajectories_dict['all_obj_trajectories']
    
    
    # check this is correct. 
    # use the phenotype clusters to label the trajectory and categorize by condition. 
    all_object_label_trajectories = SAM.get_labels_for_trajectory( all_obj_trajectories,
                                                                   clust_labels,
                                                                   all_conditions=all_treatments, 
                                                                   all_unique_conditions = np.unique(all_treatments))
                  
    """
    HMM learning on the labeled trajectories 
    """
    # over all conditions. 
    all_HMM_models, all_HMM_hidden_states_permute = SAM.fit_categoricalHMM_model_to_phenotype_cluster_label_trajectories( all_object_label_trajectories,
                                                                                                                         clust_labels,
                                                                                                                         hmm_algorithm = 'map', 
                                                                                                                         hmm_random_state = 0,
                                                                                                                         hmm_implementation='scaling')
       
    """
    Draw the HMM matrix for each condition. 
    """
    for jj in np.arange(len(uniq_treatments))[:]:
        
        treatment = uniq_treatments[jj]
        transition_matrix = all_HMM_models[jj][0] # this is already correctly ordered/permuted.
        hmm_model = all_HMM_models[jj][1]
        
        
        # check entropy of each state? 
        from scipy.stats import entropy
        
        # transition_matrix_permute_order = transition_matrix[:,all_HMM_hidden_states_permute[0]].copy()
        # transition_matrix_permute_order = transition_matrix_permute_order[all_HMM_hidden_states_permute[0]].copy()
        
        entropy_states = [entropy(ee) for ee in transition_matrix]
        
        plt.figure(figsize=(8,8))
        plt.bar(np.arange(len(entropy_states)),
                entropy_states,
                width=0.5,
                color='lightgrey',
                ec='k')
        plt.ylabel('Entropy state transition')
        plt.xlabel('State')
        plt.ylim([0,2])
        plt.savefig(os.path.join(saveplotsfolder, 
                                 'HMM_transition_drug_treatment_entropy_treatment-%s.svg' %(treatment)), dpi=300, bbox_inches='tight')
        plt.show()     
        
        
        """
        visualize the transition matrix
        """
        SAM.draw_HMM_transition_graph(transition_matrix, 
                                      ax=ax, 
                                      node_colors=clust_labels_colors, 
                                      edgescale=10, 
                                      edgelabelpos=.5, 
                                      figsize=(15,15),
                                      savefile=os.path.join(saveplotsfolder,
                                                            'HMM_transition_drug_treatment-%s.svg' %(treatment)))
        plt.show()
    

    
