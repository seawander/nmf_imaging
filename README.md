# nmf_imaging [![DOI](https://zenodo.org/badge/95447087.svg)](https://zenodo.org/badge/latestdoi/95447087)


Dimensionality reduction code for images using vectorized Nonnegative Matrix Factorization (NMF) in Python. The one dimensional vectorized NMF is proposed by Zhu ([2016](http://adsabs.harvard.edu/abs/2016arXiv161206037Z)), and the sequential construction of NMF components (i.e., sNMF) is studied by Ren et al. ([2018](http://adsabs.harvard.edu/abs/2018ApJ...852..104R)) for the application in two dimensional images in astronomy. The data imputation with missing data approach using sNMF (i.e., DI-sNMF) studied by Ren et al. ([2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200100563R/abstract)) is also supported in this package. This code takes two dimensional images as input, please refer to Zhu ([2016](http://adsabs.harvard.edu/abs/2016arXiv161206037Z)) for one dimensional data.

***Prerequisite*** to run this code: the Zhu ([2016](http://adsabs.harvard.edu/abs/2016arXiv161206037Z)) code, named ```NonnegMFPy```, can be obtained from [here](https://github.com/guangtunbenzhu/NonnegMFPy) or simply type 

```pip install NonnegMFPy``` 

in your command line. The requirements of ```NonnegMFPy``` should also be met: Python ( > 3.5.1), NumPy ( > 1.11.0), and Scipy ( > 0.17.0).

## Installation
```pip install --user -e git+https://github.com/seawander/nmf_imaging.git#egg=Package```

The above command does not require administrator access, and can be run both on one's personal desktop and on a computer cluster.

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
Since the construction of the NMF components takes a considarable amount of time, the author suggests the users contructing the components only once with ```NMFcomponents```, and use the components to model the target**s** with ```NMFmodelling```, then call the BFF subtraction described in Ren et al. ([2018](http://adsabs.harvard.edu/abs/2018ApJ...852..104R)) with ```NMFbff``` and ```NMFsubtraction```.

Example:
```python
import nmf_imaging
components = nmf_imaging.NMFcomponents(refs, ref_err = refs_err, mask = mask, n_components = componentNum, maxiters = maxiters, oneByOne=oneByOne)
#The above line construct the NMF components using the references. 
#The components can be stored in local disk to save future computational cost.

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
### 4. Data Imputation
Ignore a certain fraction of data either in component construction, or in target modeling, or both (Ren et al. [2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200100563R/abstract)).
#### 4.1 Ignore a fraction of data in component construction
Say you would like to ignore a fraction of data in component construction. Construct a 3D binary array ```mask_new``` that is of the same dimension as the references ```refs```, and make its elements to be 0 for the indices of the to-be-ignored elements (or the "missing data") in ```refs```.
```python
import nmf_imaging
components = nmf_imaging.NMFcomponents(refs, ref_err = refs_err, mask = mask_new, n_components = componentNum, maxiters = maxiters, oneByOne=oneByOne)
# Note: "mask_new" can be a three dimensional binary array that matches the size of the refs. Put 0 there for the elements you would like to ignore.
```
#### 4.2 Ignore a fraction of data in target modeling
This is needed when you have the NMF components ```components```, no matter whether they are the original ones or the ones that are from the previous approach, and would like to ingore a fraction of the target ```trg```. Mark the to-be-imputed region with a binary mask ```mask_data_imputation``` where 0 means that element is missing and 1 otherwise.

```python
model = nmf_imaging.NMFmodelling(trg = trg, trg_err = trg_err, components = components, \
		n_components = componentNum, trg_err = trg_err, mask_components=mask, \	
		maxiters=maxiters, mask_data_imputation = mask_data_imputation)
result = trg - model
```

And voilà, ```model``` contains the data imputation model, and you can remove it from the target, and investigate what is in the residual ```result```. See Ren et al. ([2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200100563R/abstract)) for an example in astronomy.
    
## References
Original sequential NMF: Ren et al. (2018), publised in the Astrophysical Journal (ADS [link](https://ui.adsabs.harvard.edu/abs/2018ApJ...852..104R/abstract)). [![DOI](https://img.shields.io/badge/DOI-10.3847/1538--4357/aaa1f2-blue)](https://doi.org/10.3847/1538-4357/aaa1f2)

Data Imputation using sequential NMF: Ren et al. (2020), published in the Astrophysical Journal (ADS [link](https://ui.adsabs.harvard.edu/abs/2020arXiv200100563R/abstract)). [![DOI](https://img.shields.io/badge/DOI-10.3847/1538--4357/ab7024-blue)](https://doi.org/10.3847/1538-4357/ab7024)




*BibTex* if you use the AASTeX package.
```
@misc{nmfimaging,
  author       = {Bin Ren},
  title        = {nmf\_imaging, doi: \href{https://doi.org/10.5281/zenodo.3738623}{10.5281/zenodo.3738623}},
  month        = apr,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v2.0},
  doi          = {10.5281/zenodo.3738623},
  url          = {https://doi.org/10.5281/zenodo.3738623}
}
```
