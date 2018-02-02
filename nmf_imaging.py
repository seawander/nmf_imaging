from NonnegMFPy import nmf
import numpy as np

def columnize(data, mask = None):
    """  Columnize an image or an image cube, excluding the masked out pixels
    Inputs:
        data: (n * height * width) or (height * width)
        mask: height * width
    Output:
        columnized: (n_pixel * n) where n_pixel is the number of unmasked pixels
    """
    if len(data.shape) == 2:
        #indicating we are flattending an image rather than a cube.
        if mask is None:
            mask = np.ones(data.shape)
        
        mask[mask < 0.9] = 0
        mask[mask != 0] = 1
        #clean the mask
        
        mask_flt = mask.flatten()
        data_flt = data.flatten()
        
        columnized = np.zeros((int(np.prod(data.shape)-np.prod(mask.shape)+np.nansum(mask)), 1))

        columnized[:, 0] = data_flt[mask_flt == 1]
        
        return columnized
        
    elif len(data.shape) == 3:
        #indicating we are vectorizing an image cube
        if mask is None:
            mask = np.ones(data.shape[1:])
        
        mask[mask < 0.9] = 0
        mask[mask != 0] = 1
        #clean the mask
        
        mask_flt = mask.flatten()
        
        columnized = np.zeros((int(np.prod(data.shape[1:])-np.prod(mask.shape)+np.nansum(mask)), data.shape[0]))
        
        for i in range(data.shape[0]):
            data_flt = data[i].flatten()
            columnized[:, i] = data_flt[mask_flt == 1]
        
        return columnized
        
def decolumnize(data, mask):
    """Decolumize either the components or the modelling result. i.e., to an image!
    data: NMF components or modelling result
    mask: must be given to restore the proper shape
    """
    mask_flatten = mask.flatten()
    
    if (len(data.shape) == 1) or (data.shape[1] == 1):
        #single column to decolumnize
        mask_flatten[np.where(mask_flatten == 1)] = data
        return mask_flatten.reshape(mask.shape)
    else:
        #several columns to decolumnize
        result = np.zeros((data.shape[1], mask.shape[0], mask.shape[1]))
        for i in range(data.shape[1]):
            results_flatten = np.copy(mask_flatten)
            results_flatten[np.where(mask_flatten == 1)] = data[:, i]
            result[i] = results_flatten.reshape(mask.shape)
            
        return result

def NMFcomponents(ref, ref_err = None, mask = None, n_components = None, maxiters = 1e3, oneByOne = False):
    """ref and ref_err should be (n * height * width) where n is the number of references. Mask is the region we are interested in.
    """
    if ref_err is None:
        ref_err = np.sqrt(ref)
    
    if mask is None:
        mask = np.ones(ref.shape[1:])
        
    if (n_components is None) or (n_components > ref.shape[0]):
        n_components = ref.shape[0]
        
    mask[mask < 0.9] = 0
    mask[mask != 0] = 1    
    
    ref[ref < 0] = 0
    ref_err[ref <= 0] = np.nanpercentile(ref_err, 95)*10 #Setting the err of <= 0 pixels to be max error to reduce their impact
    
    ref_columnized = columnize(ref, mask = mask)
    ref_err_columnized = columnize(ref_err, mask = mask)
    components_column = 0
    if not oneByOne:
        g_img = nmf.NMF(ref_columnized, V=1.0/ref_err_columnized**2, n_components=n_components)
        chi2, time_used = g_img.SolveNMF(maxiters=maxiters)
        components_column = g_img.W/np.sqrt(np.nansum(g_img.W**2, axis = 0)) #normalize the components
        components = decolumnize(components_column, mask = mask)
        
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
    
        components = decolumnize(components_column, mask = mask)
    
    return components
    
def NMFmodelling(trg, components, n_components = None, trg_err = None, mask_components = None, mask_interested = None, maxiters = 1e3, returnChi2 = False, projectionsOnly = False, coefsAlso = False, cube = False, trgThresh = 1.0):
    """
    trg: height * width
    components: n * height * width, calculated using NMFcomponents.
        mask_components: height * width, the mask used in NMFcomponents.
    n_components: how many components do you want to use. If None, all the components will be used.
    
    mask_insterested: height * width, the region you are interested in.
    projectionsOnly: output the individual projection results.
    cube: whether output a cube or not (increasing the number of components).
    trgThresh: ignore the regions with low photon counts. Especially when they are ~10^-15 or smaller. I chose 1 in this case.
    """
    if mask_interested is None:
        mask_interested = np.ones(trg.shape)
    if mask_components is None:
        mask_components = np.ones(trg.shape)
    if n_components is None:
        n_components = components.shape[0]
        
    mask = mask_components*mask_interested
    
    mask[mask < 0.9] = 0
    mask[mask != 0] = 1
        
    if trg_err is None:
        trg_err = np.sqrt(trg)
        
    trg[trg < trgThresh] = 0
    trg_err[trg == 0] = np.nanpercentile(trg_err, 95)*10
    
    components_column = columnize(components[:n_components], mask = mask)
    components_column = components_column/np.sqrt(np.nansum(components_column**2, axis = 0)) #normalize the components
    
    trg_column = columnize(trg, mask = mask)
    trg_err_column = columnize(trg_err, mask = mask)
    if not cube:
        trg_img = nmf.NMF(trg_column, V=1/trg_err_column**2, W=components_column, n_components = n_components)
        (chi2, time_used) = trg_img.SolveNMF(H_only=True, maxiters = maxiters)
    
        coefs = trg_img.H
        
        if not projectionsOnly:
            # return only the final result
            model_column = np.dot(components_column, coefs)
    
            model = decolumnize(model_column, mask)
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
    
            model_slice = decolumnize(model_column, mask)
            
            if i == 0:
                model = np.zeros((n_components, ) + model_slice.shape)
            model[i] = model_slice
            
    if returnChi2:
        return model, chi2
    if coefsAlso:
        return model, coefs
    return model
    
def NMFsubtraction(trg, model, mask = None, frac = 1):
    """Yeah subtraction!"""
    
    if mask is not None:
        trg = trg*mask
        model = model*mask
    if np.shape(np.asarray(frac)) == ():
        return trg-model*frac
    result = np.zeros((len(frac), ) + model.shape)
    for i, fraction in enumerate(frac):
        result[i] = trg-model*fraction
    return result
    
def NMFbff(trg, model, mask = None, fracs = None):
    """BFF subtraction.
    Input: trg, model, mask (if need to be), fracs (if need to be).
    Output: best frac
    """
    if mask is not None:
        trg = trg*mask
        model = model*mask
        
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
   
def nmf_func(trg, refs, trg_err = None, refs_err = None, mask = None, componentNum = 5, maxiters = 1e5, oneByOne = True, trg_type = 'disk'):
    """ Main NMF function for high contrast imaging.
    Input:  trg (2D array): target image, dimension: height * width.
            refs (3D array): reference cube, dimension: referenceNumber * height * width.
            trg_err, ref_err: uncertainty for trg and refs, repectively. If None is given, the squareroot of the two arrays will be adopted.
            mask (2D array): 0 and 1 array, the mask of the region we are interested in for NMF. 1 means the pixel we are interested in.
            componentNum (integer): number of components to be used. Default: 5. Caution: choosing too many components will slow down the computation.
            maxiters (integer): number of iterations needed. Default: 10^5.
            oneByOne (boolean): whether to construct the NMF components one by one. Default: True.
            trg_type (string): 'disk' (or 'd', for circumstellar disk) or 'planet' (or 'p', for planets). To reveal planets, the BFF procedure will not be implemented.
    Output: result (2D array): NMF modeling result. Only the final subtraction result is returned."""
    components = NMFcomponents(refs, ref_err = refs_err, mask = mask, n_components = componentNum, maxiters = maxiters, oneByOne=oneByOne)
    model = NMFmodelling(trg = trg, components = components, n_components = componentNum, trg_err = trg_err, mask_components=mask, maxiters=maxiters)
        #Bff Procedure below: for planets, it will not be implemented.
    if trg_type == 'p' or 'planet': # planets
        best_frac = 1
    elif trg_type == 'd' or 'disk': # disks
        best_frac = NMFbff(trg = sci, model = model)
    result = NMFsubtraction(trg = trg, model = model, mask = mask, frac = best_frac)
    return result
