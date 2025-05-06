import h5py
import os

import numpy as np

from typing import Tuple, Dict

def split_path(path:str):
    i = 2
    while not path[:i].endswith(".h5"):
        i += 1
    return os.path.abspath(path[:i]), path[i:]

def load(path: str, mode: str = "r"):
    """
    Loads data and arguments from .h5 file.

    Args:
        path (str): Path to .h5 file containing the data

    Returns:
        file (h5py.File): hdf5 file containing the data (given so that it can be closed if it is not necessary anymore)

        dataset (h5py.dataset): hdf5 dataset containing the data (it isn't loaded into memory automatically so that it can be handled more efficiently)
    """  
    file_path, dataset_name = split_path(path)

    file = h5py.File(file_path, mode)
    dataset = file[dataset_name]
    
    return file, dataset

def load_np(path: str) -> Tuple[np.ndarray, Dict]:
    """
    Loads data and arguments from .h5 file.

    Args:
        path (str): Path to .h5 file containing the data

    Returns:
        values (np.ndarray): hdf5 dataset containing the data 

        attributes (dict): Dictionary of metadata associated to the data
    """  
    file_path, dataset_name = split_path(path)

    file = h5py.File(file_path, "r")
    dataset = file[dataset_name]
    values = dataset[:]
    attributes = dict(dataset.attrs)
    file.close()

    return values, attributes

def save(path: str, values: np.ndarray, attributes: dict):
    """
    Saves data and arguments to .h5 file.

    Args:
        path (str): Path to .h5 file containing the data

        values (np.ndarray): Numpy array of data to be stored

        attributes (dict): Dictionary of metadata associated to the data
    """  
    file_path, dataset_name = split_path(path)

    with h5py.File(file_path, "a") as file:
        dataset = file.require_dataset(dataset_name, shape = values.shape, dtype = values.dtype)
        dataset[:] = values
        for k in attributes.keys():
            if not (attributes[k] is None):
                dataset.attrs[k] = attributes[k]

def load_source_target(source_path: str, target_path: str):
    """
    Load the source and target hdf5 files and datasets. The target file is opened in "a" mode. It is equal to the source file if the file paths are the same. Otherwise the source is opened in read mode.

    Args:
        path_source (str): Path to the source file

        path_target (str): Path to the target file

    Returns:   
        source_file (h5py.File): hdf5 file containing the data (given so that it can be closed if it is not necessary anymore)

        source_dataset (np.ndarray): hdf5 dataset containing the data (it isn't loaded into memory automatically so that it can be handled more efficiently)

        target_file (h5py.File): hdf5 file containing the data (given so that it can be closed if it is not necessary anymore)
    """

    source_file_path, source_dataset_name = split_path(source_path)
    target_file_path, target_dataset_name = split_path(target_path)

    if source_file_path != target_file_path:
        source_file = h5py.File(source_file_path, "r")
        source_configs = source_file[source_dataset_name]
        target_file = h5py.File(target_file_path, "a")
    else:
        source_file = h5py.File(source_file_path, "a")
        source_configs = source_file[source_dataset_name]
        target_file = source_file
    
    return source_file, source_configs, target_file, target_dataset_name 

def abspath(path):
    """
    Returns the absolute path of the file.
    """

    file_path, dataset_name = split_path(path)
    abs_file_path = os.path.abspath(file_path)

    return abs_file_path + dataset_name
