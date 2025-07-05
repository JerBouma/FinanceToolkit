"""Cache Module"""

__docformat__ = "google"

import os
import pickle

import pandas as pd

from financetoolkit.utilities import logger_model

logger = logger_model.get_logger()

# pylint: disable=comparison-with-itself,too-many-locals,protected-access


def load_cached_data(
    cached_data_location: str,
    file_name: str,
    method: str = "pandas",
    return_empty_type: pd.DataFrame | dict = pd.DataFrame(),
):
    """
    Load the cached data from the specified location and file name.

    Args:
        cached_data_location (str): The location of the cached data.
        file_name (str): The name of the file to load.
        method (str): The method to use for loading the data, either "pandas" or "pickle".
        return_empty_type (pd.DataFrame | dict): The type to return if the file is not found.

    Returns:
        pd.DataFrame | dict: The loaded DataFrame or dictionary.
    """
    try:
        if method == "pandas":
            cached_data = pd.read_pickle(f"{cached_data_location}/{file_name}")
        elif method == "pickle":
            with open(f"{cached_data_location}/{file_name}", "rb") as file:
                cached_data = pickle.load(file)
        else:
            raise ValueError("The method should be either 'pandas' or 'pickle'.")

        return cached_data

    except FileNotFoundError:
        return return_empty_type


def save_cached_data(
    cached_data: pd.DataFrame,
    cached_data_location: str,
    file_name: str,
    method: str = "pandas",
    include_message: bool = True,
):
    """
    Save the cached data to the specified location and file name.

    Args:
        cached_data_location (str): The location to save the cached data.
        file_name (str): The name of the file to save.
    """
    os.makedirs(cached_data_location, exist_ok=True)

    if file_name in os.listdir(cached_data_location):
        # When the file already exists do nothing.
        pass
    else:
        try:
            if method == "pandas":
                cached_data.to_pickle(f"{cached_data_location}/{file_name}")
            elif method == "pickle":
                with open(f"{cached_data_location}/{file_name}", "wb") as file:
                    pickle.dump(cached_data, file, protocol=pickle.HIGHEST_PROTOCOL)

            if include_message:
                logger.info(
                    "The data has been saved to %s/%s", cached_data_location, file_name
                )
        except Exception as error:  # pylint: disable=broad-except
            logger.error("An error occurred while saving the data: %s", error)
