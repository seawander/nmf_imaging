# This code is the nmf_imaging.py adjusted for pyKLIP, see https://bitbucket.org/pyKLIP/pyklip.
from NonnegMFPy import nmf
import numpy as np

def NMFcomponents(ref, ref_err = None, n_components = None, maxiters = 1e3, oneByOne = False):
    """Returns the NMF components, where the rows contain the information.
    Input: ref and ref_err should be (N * p) where n is the number of references, p is the number of pixels in each reference.
    Output: NMf components (n_components * p).
    """
    if ref_err is None:
        ref_err = np.sqrt(ref)
        
    if (n_components is None) or (n_components > ref.shape[0]):
        n_components = ref.shape[0]

    ref[ref < 0] = 0
    ref_err[ref <= 0] = np.nanpercentile(ref_err, 95)*10 #Setting the err of <= 0 pixels to be max error to reduce their impact
    
    ref_columnized = ref.T         #columnize ref, making the columns contain the information
    ref_err_columnized = ref_err.T # columnize ref_err, making the columns contain the information
    components_column = 0
    if not oneByOne:
        g_img = nmf.NMF(ref_columnized, V=1.0/ref_err_columnized**2, n_components=n_components)
        chi2, time_used = g_img.SolveNMF(maxiters=maxiters)
        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components        
    else:
        print("Building components one by one...")
        for i in range(n_components):
            print("\t" + str(i+1) + " of " + str(n_components))
            n = i + 1
            if (i == 0):
                g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, n_components= n)
            else:
                W_ini = np.random.rand(ref_columnized.shape[0], n)
                W_ini[:, :(n-1)] = np.copy(g_img.W)
                W_ini = np.array(W_ini, order = 'F') #Fortran ordering, column elements contiguous in memory.
                
                H_ini = np.random.rand(n, ref_columnized.shape[1])
                H_ini[:(n-1), :] = np.copy(g_img.H)
                H_ini = np.array(H_ini, order = 'C') #C ordering, row elements contiguous in memory.
                
                g_img = nmf.NMF(ref_columnized, V = 1.0/ref_err_columnized**2, W = W_ini, H = H_ini, n_components= n)
            chi2 = g_img.SolveNMF(maxiters=maxiters)
            
            components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
    
    return components_column.T
    
def NMFmodelling(trg, components, n_components = None, trg_err = None, maxiters = 1e3, returnChi2 = False, projectionsOnly = False, coefsAlso = False, cube = False, trgThresh = 1.0):
    """ NMF modeling.
    Inputs:
        trg: 1D array, p pixels
        components: N * p, calculated using NMFcomponents.
        n_components: how many components do you want to use. If None, all the components will be used.
    
        projectionsOnly: output the individual projection results.
        cube: whether output a cube or not (increasing the number of components).
        trgThresh: ignore the regions with low photon counts. Especially when they are ~10^-15 or smaller. I chose 1 in this case.
    
    Returns:
        NMF model of the target.
    """
    if n_components is None:
        n_components = components.shape[0]
        
    if trg_err is None:
        trg_err = np.sqrt(trg)
        
    trg[trg < trgThresh] = 0
    trg_err[trg == 0] = np.nanpercentile(trg_err, 95)*10
    
    components_column = components.T   #columnize the components, make sure NonnegMFPy returns correct results.
    components_column = components_column/np.sqrt(np.nansum(components_column**2, axis = 0)) #normalize the components #make sure the components are normalized.
    
    #Columnize the target and its error.
    trg_column = np.zeros((trg.shape[0], 1))
    trg_column[:, 0] = trg
    trg_err_column = np.zeros((trg_err.shape[0], 1))
    trg_err_column[:, 0] = trg_err
    if not cube:
        trg_img = nmf.NMF(trg_column, V=1/trg_err_column**2, W=components_column, n_components = n_components)
        (chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)
    
        coefs = trg_img.H
        
        if not projectionsOnly:
            # return only the final result
            model_column = np.dot(components_column, coefs)
    
        else:
            # return the individual projections
            if not coefsAlso:
                return (coefs.flatten() * components.T).T
            else:
                return (coefs.flatten() * components.T).T, coefs
    else:
        print("Building models one by one...")
        
        for i in range(n_components):
            print("\t" + str(i+1) + " of " + str(n_components))
            trg_img = nmf.NMF(trg_column, V=1/trg_err_column**2, W=components_column[:, :i+1], n_components = i + 1)
            (chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)
    
            coefs = trg_img.H
    
            model_column = np.dot(components_column[:, :i+1], coefs)
            
    if returnChi2:
        return model_column.T, chi2
    if coefsAlso:
        return model_column.T, coefs
    return model_column.T.flatten()
    
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
    Input: trg, model, fracs (if need to be).
    Output: best frac
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
   
def nmf_math(sci, ref_psfs, sci_err = None, ref_psfs_err = None, componentNum = 5, maxiters = 1e5, oneByOne = True, trg_type = 'disk'):
    """
    Main NMF function for high contrast imaging.
    Args:  
        trg (1D array): target image, dimension: height * width.
        refs (2D array): reference cube, dimension: referenceNumber * height * width.
        trg_err, ref_err: uncertainty for trg and refs, repectively. If None is given, the squareroot of the two arrays will be adopted.
    
        componentNum (integer): number of components to be used. Default: 5. Caution: choosing too many components will slow down the computation.
        maxiters (integer): number of iterations needed. Default: 10^5.
        oneByOne (boolean): whether to construct the NMF components one by one. Default: True.
        trg_type (string): 'disk' (or 'd', for circumstellar disk) or 'planet' (or 'p', for planets). To reveal planets, the BFF procedure will not be implemented.
    Returns: 
        result (1D array): NMF modeling result. Only the final subtraction result is returned.
    """
    badpix = np.where(np.isnan(sci))
    sci[badpix] = 0
    
    components = NMFcomponents(ref_psfs, ref_err = ref_psfs_err, n_components = componentNum, maxiters = maxiters, oneByOne=oneByOne)
    model = NMFmodelling(trg = sci, components = components, n_components = componentNum, trg_err = sci_err, maxiters=maxiters)
    
    #Bff Procedure below: for planets, it will not be implemented.
    if trg_type == 'p' or 'planet': # planets
        best_frac = 1
    elif trg_type == 'd' or 'disk': # disks
        best_frac = NMFbff(trg = sci, model = model)
    
    result = NMFsubtraction(trg = sci, model = model, frac = best_frac)
    return result.flatten()
