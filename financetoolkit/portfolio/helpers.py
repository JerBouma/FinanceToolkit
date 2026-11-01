"""Helpers Module"""

import re

import pandas as pd
import yaml

from financetoolkit.utilities.excel_model import read_excel as read_workbook

# pylint: disable=too-few-public-methods


class Style:
    """
    This class is meant for easier styling throughout the application where it
    adds value (e.g. to create a distinction between an error and warning).
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def read_excel(location: str):
    """
    Read an Excel (.xlsx or .xls) or CSV (.csv) file into a Pandas DataFrame.

    This function reads and loads data from an Excel or CSV file located at the specified 'location'
    into a Pandas DataFrame.

    Parameters:
        location (str): The file path of the Excel or CSV file to read.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the file.

    Raises:
        ValueError: If the specified file does not have a '.xlsx' or '.csv' extension.
    """
    if location.endswith(".xlsx") or location.endswith(".xls"):
        return read_workbook(location)
    if location.endswith(".csv"):
        csv_dataset = pd.read_csv(location, delimiter=",")

        if len(csv_dataset.columns) == 1:
            csv_dataset = pd.read_csv(location, delimiter=";")

        return csv_dataset

    raise ValueError("File type not supported. Please use .xlsx or .csv")


def read_yaml_file(location: str):
    """
    Read and parse a YAML file.

    This function reads a YAML file located at the specified 'location' and parses its contents into
    a Python dictionary. It handles exceptions for file not found, YAML parsing errors, and other
    general exceptions.

    Parameters:
        location (str): The file path of the YAML file to read and parse.

    Returns:
        dict: A dictionary containing the parsed YAML data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML content.
        Exception: For any other general exceptions that may occur during file reading or parsing.
    """
    try:
        with open(location) as yaml_file:
            data = yaml.safe_load(yaml_file)
        return data
    except FileNotFoundError as exc:
        raise ValueError(f"The file '{location}' does not exist.") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing '{location}': {exc}") from exc
    except KeyError as exc:
        raise ValueError(f"An error occurred: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"An error occurred: {exc}") from exc


def convert_to_float(value: str):
    """ "
    Convert a numeric string with locale-specific formatting to a float using a dot as the decimal separator.

    This function handles input strings formatted in one of two ways:
    - Using a dot as the thousands separator and a comma as the decimal separator (e.g., "-4.587,24").
    - Using a comma as the thousands separator and a dot as the decimal separator (e.g., "-4,587.24").

    Whole numbers with more than one thousands separator ("1,234,567" or "1.234.567") lose their
    separators, and spaces or apostrophes between the thousands are removed. A single comma is a
    decimal comma ("2,500" is 2.5, as in a semicolon-separated European file) and a single dot is
    a decimal point ("1.234" is 1.234).

    The function works as follows:
    1. If the input is a string, it checks its format using regular expressions.
    2. Depending on the detected format, it replaces the appropriate characters:
        - For numbers like "-4.587,24": removes dots (thousands separator) and replaces the comma with a dot.
        - For numbers like "-4,587.24": removes commas (thousands separator).
    3. Converts the resulting string to a float.

    If the input is not a string, it will attempt to convert it directly to float.

    Args:
        value (str): The numeric string to convert to a float.

    Raises:
         ValueError: If the conversion fails or the input does not conform to the expected numeric formats.

    Returns:
         float: The converted floating-point number.
    """
    try:
        if isinstance(value, str):
            # Spaces, non-breaking spaces and apostrophes only separate thousands ("1 234,56", "1'234.56").  # noqa: E501
            value = re.sub(r"[\s\u00a0\u202f']", "", value)
            # Match numbers like "-4.587,24", "1234,56" and "2,500": a single comma is a decimal comma.
            if re.match(r"^-?(\d{1,3}(\.\d{3})*|\d+),\d+$", value):
                value = value.replace(".", "").replace(",", ".")
            # Match numbers like "-4,587.24" and "1,234,567"
            elif re.match(r"^-?\d{1,3}((,\d{3})*\.\d+|(,\d{3}){2,})$", value):
                value = value.replace(",", "")
            # Match numbers like "1.234.567"
            elif re.match(r"^-?\d{1,3}(\.\d{3}){2,}$", value):
                value = value.replace(".", "")
        return float(value)
    except Exception as e:
        raise ValueError(f"Cannot convert {value} to float: {e}") from e
