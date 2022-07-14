# This code is the nmf_imaging.py adjusted for pyKLIP at https://bitbucket.org/pyKLIP/pyklip/src/master/pyklip/nmf_imaging.py
# Another version is kept at https://github.com/seawander/nmf_imaging/blob/master/nmf_imaging_for_pyKLIP.py

from NonnegMFPy import nmf
import numpy as np
import os
from astropy.io import fits

def data_masked_only(data, mask = None):
    """ Return the data where the same regions are ignored in all the data
    Args:
        data: (p, N) where N is the number of references, p is the number of pixels in each reference
        mask: 1d array containing only 1 and 0, with 0 will be ignored in the output
    Returns:
        data_focused: (p_focused, N) where p_focused is the number of 1's in mask
    """ 
    p_focused = int(np.nansum(mask)) # number of pixels that will be included
    
    if len(data.shape) == 2:
        data_focused = np.zeros((p_focused, data.shape[1])) # output
    
        for i in range(data.shape[1]):
            data_focused[:, i] = data[:, i][np.where(mask == 1)]
    elif len(data.shape) == 1:
        data_focused = data[mask == 1] # output
        data_focused = data_focused[:, np.newaxis]
    
    return data_focused
    
def data_masked_only_revert(data, mask = None):
    """ Return the data where the same regions were ignored in all the data
    Args:
        data: (p_focused, N) where N is the number of references, p_focused is the number of 1's in mask
        mask: 1d array containing only 1 and 0, with 0 was previously ignored in the input
    Returns:
        data_focused_revert: (p, N) where p is the number of pixels in each reference
    """ 
    data_focused_revert = np.zeros((len(mask), data.shape[1])) * np.nan
    
    for i in range(data.shape[1]):
        data_focused_revert[np.where(mask == 1), i] = data[:, i]
    
    return data_focused_revert
        
def NMFcomponents(ref, ref_err = None, n_components = None, maxiters = 1e3, oneByOne = False, ignore_mask = None, path_save = None, recalculate = False):
    """Returns the NMF components, where the rows contain the information.
    Args:
        ref and ref_err should be (N, p) where N is the number of references, p is the number of pixels in each reference.
        ignore_mask: array of shape (N, p). mask pixels in each image that you don't want to use. 
        path_save: string, path to save the NMF components (at: path_save + '_comp.fits') and coeffieients (at: path_save + '_coef.fits')
        recalculate: boolean, whether to recalculate when path_save is provided
    Returns: 
        NMf components (n_components * p).
    """
    ref = ref.T # matrix transpose to comply with statistician standards on storing data
    
    if ref_err is None:
        ref_err = np.sqrt(ref)
    else:
        ref_err = ref_err.T # matrix transpose for the error map as well
        
    if (n_components is None) or (n_components > ref.shape[0]):
        n_components = ref.shape[0]
        
    if ignore_mask is None:
        ignore_mask = np.ones_like(ref)
    
    # ignore certain values in component construction
    ignore_mask[ref <= 0] = 0 # 1. negative values
    ignore_mask[~np.isfinite(ref)] = 0 # 2. infinite values
    ignore_mask[np.isnan(ref)] = 0 # 3. nan values
    
    ignore_mask[ref_err <= 0] = 0 # 1. negative values in input error map
    ignore_mask[~np.isfinite(ref_err)] = 0 # 2. infinite values in input error map
    ignore_mask[np.isnan(ref_err)] = 0 # 3. nan values in input error map
    
    # speed up component calculation by ignoring the commonly-ignored elements across all references
    mask_mark = np.nansum(ignore_mask, axis = 1)
    mask_mark[mask_mark != 0] = 1 # 1 means that there is coverage in at least one of the refs
    
    ref_columnized = data_masked_only(ref, mask = mask_mark)
    ref_err_columnized = data_masked_only(ref_err, mask = mask_mark)
    mask_columnized = data_masked_only(ignore_mask, mask = mask_mark)
    mask_columnized_boolean = np.array(data_masked_only(ignore_mask, mask = mask_mark), dtype = bool)
    ref_columnized[mask_columnized == 0] = 0 # assign 0 to ignored values, should not impact the final result given the usage of mask_columnized_boolean
    ref_err_columnized[mask_columnized == 0] = np.nanmax(ref_err_columnized) # assign max uncertainty to ignored values, should not impact the final result

    
    # component calculation
    components_column = 0
    if not oneByOne:
        g_img = nmf.NMF(ref_columnized, V=1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components=n_components)
        chi2, time_used = g_img.SolveNMF(maxiters=maxiters)
        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components        
        components = data_masked_only_revert(components_column, mask = mask_mark)        
    else:
        print("Building components one by one...")
        if path_save is None or recalculate:
            if recalculate:
                print('Recalculating no matter if you have saved previous ones.')
            for i in range(n_components):
                print("\t" + str(i+1) + " of " + str(n_components))
                n = i + 1
                if (i == 0):
                    g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                else:
                    W_ini = np.random.rand(ref_columnized.shape[0], n)
                    W_ini[:, :(n-1)] = np.copy(g_img.W)
                    W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
                
                    H_ini = np.random.rand(n, ref_columnized.shape[1])
                    H_ini[:(n-1), :] = np.copy(g_img.H)
                    H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
                
                    g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, W = W_ini, H = H_ini, n_components= n)
                chi2 = g_img.SolveNMF(maxiters=maxiters)
            
                components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                components = data_masked_only_revert(components_column, mask = mask_mark) 
            if recalculate:
                print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
        else:
            if not os.path.exists(path_save + '_comp.fits'):
                print('\t\t ' + path_save + '_comp.fits does not exist, calculating from scratch.')
                for i in range(n_components):
                    print("\t" + str(i+1) + " of " + str(n_components))
                    n = i + 1
                    if (i == 0):
                        g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, n_components= n)
                    else:
                        W_ini = np.random.rand(ref_columnized.shape[0], n)
                        W_ini[:, :(n-1)] = np.copy(g_img.W)
                        W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
                
                        H_ini = np.random.rand(n, ref_columnized.shape[1])
                        H_ini[:(n-1), :] = np.copy(g_img.H)
                        H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
                
                        g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, M = mask_columnized_boolean, W = W_ini, H = H_ini, n_components= n)
                    chi2 = g_img.SolveNMF(maxiters=maxiters)
                    print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                    fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                    print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                    fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
                    components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                    components = data_masked_only_revert(components_column, mask = mask_mark)
            else:
                W_assign = fits.getdata(path_save + '_comp.fits')
                H_assign = fits.getdata(path_save + '_coef.fits')
                if W_assign.shape[1] >= n_components:
                    print('You have already had ' + str(W_assign.shape[1]) + ' components while asking for ' + str(n_components) + '. Returning to your input.')
                    components_column = W_assign/np.sqrt(np.nansum(W_assign**2, axis = 0))
                    components = data_masked_only_revert(components_column, mask = mask_mark)
                else:
                    print('You are asking for ' + str(n_components) + ' components. Building the rest based on the ' + str(W_assign.shape[1]) + ' provided.')

                    for i in range(W_assign.shape[1], n_components):
                        print("\t" + str(i+1) + " of " + str(n_components))
                        n = i + 1
                        if (i == W_assign.shape[1]):
                            W_ini = np.random.rand(ref_columnized.shape[0], n)
                            W_ini[:, :(n-1)] = np.copy(W_assign)
                            W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                            H_ini = np.random.rand(n, ref_columnized.shape[1])
                            H_ini[:(n-1), :] = np.copy(H_assign)
                            H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                            g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, M = mask_columnized_boolean, n_components= n)
                        else:
                            W_ini = np.random.rand(ref_columnized.shape[0], n)
                            W_ini[:, :(n-1)] = np.copy(g_img.W)
                            W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
            
                            H_ini = np.random.rand(n, ref_columnized.shape[1])
                            H_ini[:(n-1), :] = np.copy(g_img.H)
                            H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
            
                            g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, M = mask_columnized_boolean, n_components= n)
                        chi2 = g_img.SolveNMF(maxiters=maxiters)
                        print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D component matrix at ' + path_save + '_comp.fits')
                        fits.writeto(path_save + '_comp.fits', g_img.W, overwrite = True)
                        print('\t\t\t Calculation for ' + str(n) + ' components done, overwriting raw 2D coefficient matrix at ' + path_save + '_coef.fits')
                        fits.writeto(path_save + '_coef.fits', g_img.H, overwrite = True)
                        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
                        components = data_masked_only_revert(components_column, mask = mask_mark)            
    return components.T
    
def NMFmodelling(trg, components, n_components = None, mask_components = None, mask_data_imputation = None, trg_err = None, maxiters = 1e3, cube = False, trgThresh = 0):
    """ NMF modeling.
    Args:
        trg: 1D array, p pixels
        components: N * p, calculated using NMFcomponents.
        n_components: how many components do you want to use. If None, all the components will be used.
        cube: whether output a cube or not (increasing the number of components).
        trgThresh: ignore the regions with low photon counts. Especially when they are ~10^-15 or smaller. I chose 0 in this case.
    
    Returns:
        NMF model of the target.
    """

    
    if n_components is None:
        n_components = components.shape[0]
        
    if trg_err is None:
        trg_err = np.sqrt(trg)

    if mask_components is None:
        mask_components = np.ones(trg.shape)
        mask_components[np.where(np.isnan(components[0]))] = 0

    components_column_all = data_masked_only(components[:n_components].T, mask = mask_components)   #columnize the components, make sure NonnegMFPy returns correct results.
    components_column_all = components_column_all/np.sqrt(np.nansum(components_column_all**2, axis = 0)) #normalize the components #make sure the components are normalized.
    
    if mask_data_imputation is None:
        flag_di = 0
        mask_data_imputation = np.ones(trg.shape)
    else:
        flag_di = 1
        print('Data Imputation!')
        
    mask = mask_components*mask_data_imputation #will be used for modeling

    trg[trg < trgThresh] = 0
    trg_err[trg == 0] = np.nanmax(trg_err)

    mask[trg <= 0] = 0
    mask[np.isnan(trg)] = 0
    mask[~np.isfinite(trg)] = 0
    
    #Columnize the target and its error.
    trg_column = data_masked_only(trg, mask = mask)
    trg_err_column = data_masked_only(trg_err, mask = mask)
    components_column = data_masked_only(components.T, mask = mask)

    if not cube:
        trg_img = nmf.NMF(trg_column, V=1/trg_err_column**2, W=components_column, n_components = n_components)
        (chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)
        coefs = trg_img.H
        if flag_di == 0: # do not do data imputation
            model_column = np.dot(components_column, coefs)

            model = data_masked_only_revert(model_column, mask)
            model[np.where(mask == 0)] = np.nan
        elif flag_di == 1: # do data imputation
            model_column = np.dot(components_column_all, coefs)
            model = data_masked_only_revert(model_column, mask_components)
            model[np.where(mask_components == 0)] = np.nan
    else:
        print("Building models one by one...")

        for i in range(n_components):
            print("\t" + str(i+1) + " of " + str(n_components))
            trg_img = nmf.NMF(trg_column, V=1/trg_err_column**2, W=components_column[:, :i+1], n_components = i + 1)
            (chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)

            coefs = trg_img.H

            model_column = np.dot(components_column[:, :i+1], coefs)

    return model.flatten() #model_column.T.flatten()
    
def NMFsubtraction(trg, model, frac = 1):
    """NMF subtraction with a correction factor, frac."""
    if np.shape(np.asarray(frac)) == ():
        return (trg-model*frac).flatten()
    result = np.zeros((len(frac), ) + model.shape)
    for i, fraction in enumerate(frac):
        result[i] = trg-model*fraction
    return result
    
def NMFbff(trg, model, fracs = None):
    """BFF subtraction.
    Args:
        trg:
        model:
        fracs: (if need to be).
    Returns: 
        best frac
    """
    
    if fracs is None:
        fracs = np.arange(0.60, 1.001, 0.001)
    
    std_infos = np.zeros(fracs.shape)
    
    for i, frac in enumerate(fracs):
        data_slice = trg - model*frac
        while 1:
            if np.nansum(data_slice > np.nanmedian(data_slice) + 3*np.nanstd(data_slice)) == 0 or np.nansum(data_slice < np.nanmedian(data_slice) -10*np.nanstd(data_slice)) == 0:
                break
            data_slice[data_slice > np.nanmedian(data_slice) + 3*np.nanstd(data_slice)] = np.nan
            data_slice[data_slice < np.nanmedian(data_slice) - 10*np.nanstd(data_slice)] = np.nan
        std_info = np.nanstd(data_slice)
        std_infos[i] = std_info
    return fracs[np.where(std_infos == np.nanmin(std_infos))]   
   
def nmf_math(sci, ref_psfs, sci_err = None, ref_psfs_err = None, componentNum = 5, maxiters = 1e5, oneByOne = True, trg_type = 'disk',
            ignore_mask = None, path_save = None, recalculate = False, 
            mask_data_imputation = None):
    """
    Main NMF function for high contrast imaging.
    Args:  
        trg (1D array): target image, dimension: height * width.
        refs (2D array): reference cube, dimension: referenceNumber * height * width.
        trg_err, ref_err: uncertainty for trg and refs, repectively. If None is given, the squareroot of the two arrays will be adopted.
    
        componentNum (integer): number of components to be used. Default: 5. Caution: choosing too many components will slow down the computation.
        maxiters (integer): number of iterations needed. Default: 10^5.
        oneByOne (boolean): whether to construct the NMF components one by one. Default: True.
        trg_type (string,  default: "disk" or "d" for circumsetllar disks by Bin Ren, the user can use "planet" or "p" for planets): are we aiming at finding circumstellar disks or planets?
    Returns: 
        result (1D array): NMF modeling result. Only the final subtraction result is returned.
    """
    badpix = np.where(np.isnan(sci))
    sci[badpix] = 0

    components = NMFcomponents(ref_psfs, ref_err = ref_psfs_err, n_components = componentNum, maxiters = maxiters, oneByOne=oneByOne,
                                ignore_mask = ignore_mask, path_save = path_save, recalculate = recalculate)
                                
    if mask_data_imputation is None:
        model = NMFmodelling(trg = sci, components = components, n_components = componentNum, trg_err = sci_err, maxiters=maxiters,
                                mask_data_imputation = mask_data_imputation)

        if trg_type == "planet" or trg_type == "p":
            best_frac = 1
        elif trg_type == "disk" or trg_type == "d":
            best_frac = NMFbff(trg = sci, model = model)
        
        result = NMFsubtraction(trg = sci, model = model, frac = best_frac)
        result = result.flatten()
        result[badpix] = np.nan

    else:
        model = NMFmodelling(trg = sci, components = components, n_components = componentNum, trg_err = sci_err, maxiters=maxiters,
                                mask_data_imputation = mask_data_imputation)
                                
        result = sci - model

    return result
