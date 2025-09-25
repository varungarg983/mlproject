import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error = error
        self.error_detail = error_detail

    def __str__(self):
        return error_message_detail(self.error, self.error_detail)
    
