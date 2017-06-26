# nmf_imaging

Postprocessing Code for High Contrast Imaging using Vectorized Nonnegative Matrix Factorization (NMF) . The vectorized NMF is proposed by Zhu ([2016](http://adsabs.harvard.edu/abs/2016arXiv161206037Z)), and is studied by Ren et al. (2017) for the high contrast imaging in exoplanetary science. This code depends on the Zhu ([2016](http://adsabs.harvard.edu/abs/2016arXiv161206037Z)) [code](https://github.com/seawander/NonnegMFPy).


## Running the code:

### 1. Simplest case: 
Given a target ```trg``` (height * width), a reference cube ```refs``` (n_ref * height * width), and a 0-1 mask ```mask``` (height * width), the function ```nmf_func``` will first construct the NMF components, then model the target with the components, and finally return the BFF subtraction result (i.e., only the structures that cannot be modeled by the NMF components). 

Example:
```python
import nmf_imaging
result = nmf_imaging.nmf_func(trg = trg, refs = refs, mask = mask)
```

### 2. Simplest case with uncertainties: 
The ```trg``` and ```refs``` can be accompanied with their uncertainties (```trg_err``` and ```refs_err```) to handle the heteroscedastic uncertainties and missing data, then the above code becomes
```python
import nmf_imaging
result = nmf_imaging.nmf_func(trg = trg, refs = refs, trg_err = trg_err, refs_err = refs_err, mask = mask)
```


### 3. Expert coding with a number of targets:
Since the construction of the NMF components takes a considarable amount of time, the author suggests the users contructing the components only once with ```NMFcomponents```, and use the components to model the target**s** with ```NMFmodelling```, then call the BFF subtraction described in Ren et al. (2017) with ```NMFbff``` and ```NMF subtraction```.

Example:
```python
import nmf_imaging
components = nmf_imaging.NMFcomponents(refs, ref_err = refs_err, mask = mask, n_components = componentNum, maxiters = maxiters, oneByOne=oneByOne)
#The above line construct the NMF components using the references. The components can be stored in local disk to save future computational cost.

#Next: modeling a number of targets (especially, many exposures of a single targets):
results = np.zeros(trgs.shape) # Say trgs is a 3D array containing the targets that need NMF modeling, then results store the NMF subtraction results.

for i in range(trgs.shape[0]):
    trg = trgs[i]
    trg_err = trgs_err[i]
    model = nmf_imaging.NMFmodelling(trg = trg, trg_err = trg_err, components = components, n_components = componentNum, trg_err = trg_err, mask_components=mask, maxiters=maxiters) # Model the target with the constructed components.
    best_frac =  nmf_imaging.NMFbff(trg, model, mask) # Perform BFF procedure to find out the best fraction to model the target.
    result = nmf_imaging.NMFsubtraction(trg, model, mask, frac = best_frac) # Subtract the best model from the target
    results[i] = result

# Now `results' stores the NMF subtraction results of the targets.
```

## Dependences
NonnegMFPy, which can be obtained from [https://github.com/guangtunbenzhu/NonnegMFPy](https://github.com/guangtunbenzhu/NonnegMFPy), and its dependences: Python ( > 3.5.1), NumPy ( > 1.11.0), and Scipy ( > 0.17.0).
    
