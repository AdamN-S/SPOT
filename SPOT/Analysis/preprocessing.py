# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:30:30 2023

@author: S205272
"""
import numpy as np

def scale_normalize_curvature_features(all_feats, feature_names):
    r""" Curvature features are not invariant to size of the object. It is of units length^-1. In order to allow like-wise comparison across different objects, we multiply features by the equivalent diameter of the object. 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with size-corrected curvature features
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature. It is identical to the input and is output for consistent api
    
    """
    
    all_feats_corrected = all_feats.copy()
    
    # correction for curvature features. 
    for feat_ii, feature_name in enumerate(feature_names):
        if 'curvature' in feature_name:
            all_feats_corrected[:,feat_ii] = all_feats[:,feat_ii] * all_feats[:,feature_names == 'equivalent_diameter'].ravel()

    return all_feats_corrected, feature_names
    
    
# remove entirely zero-valued features
def remove_zero_features(all_feats, feature_names, return_index=False):
    r""" remove any features that are zero-valued across all object instances as well as any that have zero standard deviation across all object instances as they are uninformative 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with zero-valued features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature
    
    """
    import numpy as np 
    
    zero_feats = np.nanmax(all_feats, axis=0) == 0    
        
    if return_index:
        select_index = np.arange(len(zero_feats))
        select_index = select_index[~zero_feats].copy()
        
    feature_names_corrected = feature_names[~zero_feats].copy()
    all_feats_corrected = all_feats[:, ~zero_feats].copy() 
    
    std_feats = np.std(all_feats_corrected, axis=0)
    zero_std_feats = std_feats == 0
    
    if return_index:
        select_index = select_index[~zero_std_feats].copy()
        
    feature_names_corrected = feature_names_corrected[~zero_std_feats]
    all_feats_corrected = all_feats_corrected[:, ~zero_std_feats].copy() # so we have 400 feats. 
    
    if return_index:
        return all_feats_corrected, feature_names_corrected, select_index
    else:
        return all_feats_corrected, feature_names_corrected


# remove entirely zero-valued features
def remove_nan_variance_features(all_feats, feature_names, return_index=False):
    r""" remove any features that have zero variation across all object instances as well as any that have zero standard deviation across all object instances as they are uninformative 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with zero-valued features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature
    
    """
    import numpy as np 
    
    zero_feats = np.isnan(np.std(all_feats, axis=0))     
        
    if return_index:
        select_index = np.arange(len(zero_feats))
        select_index = select_index[~zero_feats].copy()
    
    feature_names_corrected = feature_names[~zero_feats].copy()
    all_feats_corrected = all_feats[:, ~zero_feats].copy() 
    
    if return_index:
        return all_feats_corrected, feature_names_corrected, select_index
    else:
        return all_feats_corrected, feature_names_corrected

    
def remove_high_variance_features(all_feats, feature_names, variance_threshold_sigma=3, return_index=False):
    r""" remove features that are highly variable and likely to represent outliers
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with high variance features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
        
    """
    import numpy as np 
    
    high_var_feats_thresh = np.mean(np.std(all_feats, axis=0)) + variance_threshold_sigma*np.std(np.std(all_feats, axis=0))
    keep_feats = np.std(all_feats, axis=0)<=high_var_feats_thresh
    
    if return_index:
        select_index = np.arange(len(keep_feats))
        select_index = select_index[keep_feats].copy()
    # high_var_feats_thresh = np.mean(np.std(all_feats, axis=0)) + 2*np.std(np.std(all_feats, axis=0))
    # feature_names = feature_names[np.std(all_feats, axis=0)<=high_var_feats_thresh]
    
    feature_names_corrected = feature_names[keep_feats].copy()
    all_feats_corrected = all_feats[:, keep_feats].copy() 
        
    if return_index:
        return all_feats_corrected, feature_names_corrected, select_index
    else:
        return all_feats_corrected, feature_names_corrected


def select_time_varying_features(all_feats, feature_names, all_time, ridge_alpha=1., norm_time=True):
    
    r""" Selects features that are most strongly correlated with time using linear ridge regression. Features are kept if the absolute value of their regression coefficient is greater than the mean across all features
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    all_time : (n_objects, n_feats) np.array
        numpy array of the timepoint each object instance was extracted from. 
    ridge_alpha : non-negative float
        Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).
    norm_true : bool
        if True, assumes time is given as frame number and performs (all_time-1.)/(np.max(all_time-1.)) to rescale to 0-1 float
                                                                                                                               
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with high variance features removed
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
    coeffs : (n_feats,) np.array
        fitted Ridge regression coefficients. The larger the absolute magnitude, the more the feature covaries with time.
    """
    
    # what about straightforward regression of features with time?
    from sklearn import linear_model
    import numpy as np 
    
    reg = linear_model.Ridge(alpha=1.)

    if norm_time:
        all_time_score = all_time-1.
        all_time_score = all_time_score/all_time_score.max() # to normalize. 
    else:
        all_time_score = all_time.copy()
        
    reg.fit(all_feats, all_time_score)
    coeffs = reg.coef_.copy()
    coef_thresh = np.mean(np.abs(coeffs)) # thresh by the mean 
    
    keep_feats = np.abs(reg.coef_)>=coef_thresh # create a binary indicator 
    
    feature_names_corrected = feature_names[keep_feats].copy()
    all_feats_corrected = all_feats[:, keep_feats].copy() 
    
    return all_feats_corrected, feature_names_corrected, coeffs


def kernel_dim_reduction_ECC_features(all_feats, feature_names, n_dim=100, gamma=None, random_state=1, return_index=False):
    r""" finds Euler Characteristic Curve features and using Nystroem with RBF kernel to perform dimensional reduction to the specified dimension size. 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with transformed kernel ECC features
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
    ECT_rbf_feature_tformer : Scikit-learn Nystroem transformer object
        the fitted sklearn.kernel_approximation.Nystroem object
        
    """
    from sklearn.kernel_approximation import Nystroem
    import numpy as np 
    
    # drop all ECC/ECT feats
    ECT_feats_index = np.arange(all_feats.shape[1])[np.hstack(['ECT_' in name for name in feature_names])]
    non_ECT_feats_index = np.setdiff1d(np.arange(all_feats.shape[1]), ECT_feats_index)

    
    # set up the transformer
    ECT_rbf_feature_tformer = Nystroem(n_components=n_dim, 
                                       gamma=gamma,
                                       random_state=random_state) # how many features? -> say 100 for now anyway. 
    ECT_rbf_feats = ECT_rbf_feature_tformer.fit_transform(all_feats[:,ECT_feats_index])
    
    # now modify 
    all_feats_corrected = all_feats[:,non_ECT_feats_index].copy() # keep as before. 
    feature_names_corrected = feature_names[non_ECT_feats_index].copy()
    
    # add in the transformed kernel features.
    all_feats_corrected = np.hstack([all_feats_corrected, 
                                     ECT_rbf_feats])
    feature_names_corrected = np.hstack([feature_names_corrected,
                                         ['ECT_rbf_%d'%(dd+1) for dd in np.arange(ECT_rbf_feats.shape[1])]])
    
    if return_index:
        return all_feats_corrected, feature_names_corrected, ECT_rbf_feature_tformer, (non_ECT_feats_index, ECT_feats_index)    
    else:
        return all_feats_corrected, feature_names_corrected, ECT_rbf_feature_tformer
    
    
def kernel_dim_reduction_ECC_features_with_transform(all_feats, indices, ECT_rbf_feature_tformer):
    r""" finds Euler Characteristic Curve features and using Nystroem with RBF kernel to perform dimensional reduction to the specified dimension size. 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    indices : ((M,), (N,)) np.array
        non ECC features index, ECC features index
    ECT_rbf_feature_tformer : Scikit-learn Nystroem transformer object
        the fitted sklearn.kernel_approximation.Nystroem object
    
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with transformed kernel ECC features
        
    """
    import numpy as np 
    
    non_ECT_feats_index, ECT_feats_index = indices
    
    ECT_rbf_feats = ECT_rbf_feature_tformer.transform(all_feats[:,ECT_feats_index])
    
    # now modify 
    all_feats_corrected = all_feats[:,non_ECT_feats_index].copy() # keep as before. 
    
    # add in the transformed kernel features.
    all_feats_corrected = np.hstack([all_feats_corrected, 
                                     ECT_rbf_feats])
    
    return all_feats_corrected

    
    
# individual feature normalization. 
def power_transform_and_scale_features(all_feats, feature_names, all_conditions):
    r""" Applies standard scaling (zscore) followed by power transform using Yeo-Johnson to normalize, whiten features and make more Gaussian-like for each unique condition to control 
    
    Parameters
    ----------
    all_feats : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be analyzed. 
    feature_names : (n_feats,) np.array
        numpy array of the names of each computed feature
    all_conditions : integer or categorical np.array
        each unique integer corresponds to a unique condition. Within each condition we fit a different power transformer and standard scale transformer
        
    Returns
    -------
    all_feats_corrected : (n_objects, n_feats) np.array
        numpy array of all features with transformed kernel ECC features
    feature_names_corrected : (n_feats,) np.array
        numpy array of the names of each retained feature. 
    pow_tformer_dict : dict
        * 'all_obj_trajectories': list
                This is a list of all the reconstructed trajectories. Each trajectory is given as a list of the input indices that make it up. 
        * 'all_obj_trajectories_times': list
                This is a list of all the reconstructed trajectories. Each trajectory is a list of the frame number that the object instances occurred at 
    std_tformer_dict : dict
        * str(condition) : sklearn.preprocessing StandardScaler object
                Fitted StandardScaler object for the given unique condition 
        * 'all_obj_trajectories_times': list
                This is a list of all the reconstructed trajectories. Each trajectory is a list of the frame number that the object instances occurred at 
    """
    # =============================================================================
    #   Now we do individual feature normalisation -> modify in place. 
    # =============================================================================
    import numpy as np 
    from sklearn.preprocessing import PowerTransformer, StandardScaler
    
    all_feats_corrected = np.zeros(all_feats.shape)
    feature_names_corrected = feature_names.copy()
    
    pow_tformer_dict = {}
    std_tformer_dict = {}
    
    # normalizing over each condition 
    for uniq_condition in np.unique(all_conditions):
        
        select = all_conditions == uniq_condition
        
        pow_tformer = StandardScaler()
        std_tformer = PowerTransformer()
        
        feats_norm = pow_tformer.fit_transform(std_tformer.fit_transform(all_feats[select])) 
                            
        pow_tformer_dict[str(uniq_condition)] = pow_tformer
        std_tformer_dict[str(uniq_condition)] = std_tformer
        pow_tformer_dict[str(uniq_condition)+'_data_select'] = select
        std_tformer_dict[str(uniq_condition)+'_data_select'] = select
        
        all_feats_corrected[select] = feats_norm.copy()
        
    return all_feats_corrected, feature_names_corrected, pow_tformer_dict, std_tformer_dict 
    
    
def preprocess_feats(X, std_tformer, pow_tformer):
    r""" Applies already fitted scikit-learn standard scaler and powertransformer objects to the input matrix. The power transform is applied first then standard scale. 

    Parameters
    ----------
    X : (n_objects, n_feats) np.array
        numpy array of all features for all object instances to be transformed 
    std_tformer : sklearn.preprocessing.StandardScaler object
    pow_tformer : sklearn.preprocessing.power_transform object
        
    Returns
    -------
    X_tfm : (n_objects, n_feats) np.array
        numpy array of all features with transformed features
    """
    # apply learnt transformations to input features X.
    X_tfm = std_tformer.transform(pow_tformer.transform(X))

    return X_tfm 


# def map_intensity_interp2(query_pts, grid_shape, I_ref, method='spline', cast_uint8=False, s=0):

#     import numpy as np 
#     from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator 
    
#     if method == 'spline':
#         spl = RectBivariateSpline(np.arange(grid_shape[0]), 
#                                   np.arange(grid_shape[1]), 
#                                   I_ref,
#                                   s=s)
#         I_query = spl.ev(query_pts[...,0], 
#                          query_pts[...,1])
#     else:
#         spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
#                                        np.arange(grid_shape[1])), 
#                                        I_ref, method=method, bounds_error=False, fill_value=0)
#         I_query = spl((query_pts[...,0], 
#                        query_pts[...,1]))

#     if cast_uint8:
#         I_query = np.uint8(I_query)
    
#     return I_query
