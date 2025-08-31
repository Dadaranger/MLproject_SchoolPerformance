# Import required modules
import sys  # For accessing system-specific parameters and functions
from src.logger import logging  # Import logging functionality from our custom logger module

def error_message_detail(error, error_detail: sys):
    """
    Generate a detailed error message with file name, line number, and error description.
    
    Args:
        error: The error message or exception object
        error_detail: sys module to access traceback information
    
    Returns:
        str: Formatted error message with file name, line number, and error description
    """
    # Get the traceback information from sys.exc_info()
    # exc_info() returns (type, value, traceback), we only need the traceback (third element)
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract the filename where the error occurred from the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format the error message with file name, line number, and the error description
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    """
    Custom exception class for handling and formatting exceptions in the ML project.
    Inherits from the base Exception class and adds detailed error information.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the CustomException with detailed error information.
        
        Args:
            error_message: The basic error message
            error_detail: sys module for accessing traceback information
        """
        # Call the parent class (Exception) constructor
        super().__init__(error_message)
        # Generate and store the detailed error message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        """
        String representation of the exception.
        
        Returns:
            str: The detailed error message
        """
        return self.error_message
    


        