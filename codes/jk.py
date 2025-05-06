import numpy as np
from typing import List

# Resampling of data and functions of said data

def cut_configs(configs: np.ndarray, num_samples: int):
    # Eliminate beginning of configurations to have a number of configurations divisible by num_samples
    num_configs = configs.shape[0]
    num_bins = int(num_configs/num_samples)
    num_configs = num_bins*num_samples

    return configs[-num_configs:]

def resampling(O: np.ndarray, num_samples: int = 1) -> np.ndarray:
    """
    JK resampling of a set of samples O.
    
    Creates a new set of data, the jk set, from a given set of data. Each element in this set corresponds to a bin in the original set. It is obtained by removing said bin from the data set and computing the mean of the resulting set. The jk set has the same mean as the original set. However, we can compute functions of said mean by computing the functions of each element of the jackknifed set and then taking the mean of that. The error of said function can be readily computed from the standard deviation of this, bypassing thus the need of Gaussian propagation of errors. 
    
    Args:
        O (np.ndarray): Original set of data. It is expected to be a NumPy array whose 0-th index labels the elements of the set. The number of elements of the set should be divisible by num_samples. Otherwise, the minimum number of elements at the start of O are thrown out to guarantee that this condition is satisfied.
        num_samples (int): Number of samples in each bin. Should be a positive integer. Defaults to 1.
        
    Returns:
        O_jk (np.ndarray): JK set of data. It is a NumPy array whose 0-th index labels the bins of the data.
        
    Notes: It is often useful to not store the full set of data of the original set. Instead one can store the mean of each bin of the data. The jk set can then be obtained by applying this function to these means without further binning.
    """
    if O.shape[0] == 1:

        return O

    else:

        if O.shape[0] % num_samples != 0: 
            O_cut = cut_configs(O, num_samples)
        else:
            O_cut = O
    
        num_bins = O.shape[0] // num_samples

        O_jk = np.zeros((num_bins, *O.shape[1:]), dtype = O.dtype)
    
        for bin in range(num_bins):
                O_jk[bin] = np.mean(np.append(O_cut[:bin*num_samples], O_cut[(bin+1)*num_samples:], axis = 0), axis = 0)

        return O_jk

def function_resampling(operators: dict, f, range_dims: np.ndarray, type = complex, num_samples: int = 1, notification_rate: int = 1) -> np.ndarray:
    """
    Jackknife resampling of a function on a set of data. 

    Computes the jk resampled data of a function f with range np.ndarrays of dimensions range_dims of some samples O. It does so by first taking a uniform partition mathcal{P} into bins of size num_samples and considering only the samples within a bin Binmathcal{P} at a given time. It then computes the function for each sample within the bin {f(s)}_{sin B} and their average f_B := frac{1}{N}sum_{s in B}f(s). Finally, it takes the jk resampling {F_B}_{B in mathcal{P}} of {f_B}_B with 1 sample per bin, i.e. F_B :=frac{1}{|mathcal{P}|-1}sum_{B'inmathcal{P}setminus {B}}f_{B'}. This is of course, the jk resampling of O with bin size N. However, doing it this way does not require one to store {f(O_i)}_{iin X} all at the same time. 

    Args:
        operators (dict): Dictionary of {name: hdf5 dataset} of operators whose 0-th axis should index the different samples.
        
        f: Function that will be computed on each sample of operators.

        range_dims (np.ndarray): Dimensions of f[s] for any samples in operators.

        type: Data type of f[s] for any s in O. Defaults to complex.

        num_samples: Number of samples in each bin of the jk_resampling

        notification_rate: Number of bins after which a notification is printed. Defaults to 1.

    Returns:
        O_jk (np.ndarray): Array containing the jk resampled data
    """
    # Extract shape from a test operator

    test_operator = operators[list(operators.keys())[0]]
    num_configs = test_operator.shape[0]

    num_bins = num_configs//num_samples 
    means_per_bin = np.zeros((num_bins, *range_dims), type)

    for bin in range(num_bins):
        for config in range(num_samples):
            means_per_bin[bin] += f({name: operator[bin*num_samples + config] for name, operator in operators.items()})
        means_per_bin[bin] /= num_samples
        if bin % notification_rate == 0:
            print(f"finished computing {f.__name__} for bin {bin}")

    return resampling(means_per_bin)

# Compute jackknife means and errors

def mean(O: np.ndarray, err: bool = True):
    """
    Computes the mean and error of the mean of a set of data from its jk resampling.

    Args:
        O (np.ndarray): Array of jk resampled data whose 0-th axis should index the different samples.
        err (bool): Whether we want the jackknife error to be returned

    Returns:
        mean (np.ndarray): Mean of the data.
        error (np.ndarray): Error of the mean of the data.
    """
    mean = np.mean(O, axis = 0)
    if err == False:
        return mean
    else:
        error = np.sqrt((O.shape[0] - 1)*np.var(O, axis = 0))
        return mean, error

# Convenient functions when dealing with jk data

def function(f, data_jk, data_stat):
    """
    Computes the values of a function f on jackknife data

    Args:
        data_jk (list): list of numpy arrays of the data to be computed. The first index of each array must be the bin index.

        data_stat (list): list of stationary data

        f: function to be computed. It should take in first the data in data_jk. It should then take the data in data_stat. For now the output of the function must be a single variable
    """
    num_bins = len(data_jk[0])
    result = [] 

    for bin in range(num_bins):
        input = [data[bin] for data in data_jk]
        if data_stat != None:
            result.append(f(*input,*data_stat))
        else:
            result.append(f(*input))

    return np.array(result)