#%% Imports
import sys, os, time
from typing import Union
from collections.abc import Callable
import smtplib, ssl
import math

# Other
from cryptography.fernet import Fernet

#%% Logging
import logging, os
from datetime import datetime
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG) # Sets level at which info will be captured, can elevate level for CLI and file output to filter out lower level messages
formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d - %H:%M:%S")

# Logging CLI output stream
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO) # Set to INFO to display only up to INFO level
ch.setFormatter(formatter)
LOG.addHandler(ch)

# Logging file output stream
date_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
if os.path.exists("archive/logs/"):
    log_dir = "archive/logs/"
else:
    log_dir = "" # Use root dir
# fh = logging.FileHandler(F"{log_dir}{date_time}.log", "w")
fh = logging.FileHandler(F"{log_dir}{date_time}.log", "w")
fh.setLevel(logging.DEBUG) # Log info all the way down to DEBUG level  
fh.setFormatter(formatter)
LOG.addHandler(fh)
#%% CONSTANTS

ARGS = sys.argv # Container for CLI arguments, probably wont be used here since this is not an entry point

#%% Functions

def waitTime(duration: Union[int, float],
             fx: Callable = None,
             loop_delay: float = 60,
             verbose: bool = True,
             **kwargs):
    """Wait for a specified duration while optionally looping through a 
    given function

    Args:
        duration (Union[int, float]): Durtaion to wait in MINUTES
        fx (Callable, optional): Function to repeat over. This fx must RETURN TRUE to exit. Defaults to None.
        loop_delay (float, optional): SECONDS between each loop iteration
        verbose (bool, optional): Whether time reports will be printed. Defaults to True.
    """
    time_end = time.time() + float(duration*60) # Time will be converted from minutes to seconds 
    while time.time() < time_end:
        if fx: # If there is a fx provided
            output = fx(**kwargs)
            if output:
                print(F"Function condition satisfied, exiting wait early")
                return
        if verbose:
            print(f"Time left (minutes): {str(math.ceil((time_end-time.time()))/60)}")
        time.sleep(loop_delay)

    

#%% Classes


class GeneralCVBot:
    """
    Parts to general purpose CV bot:
        -Wait time before execution
        -Duration of execution
        -Root image name
        -Number of images to iterate through 
        -Which screen
        -Action to do 
        -Frequency of checking 
    """
    pass

class Cryptographer:
    """
    Class containiner cryptography functions
    
    Note that you can rerun encrypt step to add multiple layers of 
    encryption which will need subsequent decryption layers with the corresponding
    key that was used for decryption
    """

    def __init__(self):
        self.key = bytes() # Empty bytestring to receive key when generated
        self.fernet = None
        
    def genKey(self):
        """
        Generates a key and initializes Fernet cryptographer with it
        """
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)
        return self
        
    def exportKey(self, path: Union[str, bytes, os.PathLike] = "data/fernetkey") -> bytes:
        """
        Exports generated key to a given file

        Args:
            path: Target export path for key

        Returns:
            Exported key in bytes
        """
        if self.key: # If key exists:
            with open(path, "w+b") as file: # Need to open in binary mode
                # w+ parameter specifies to create file if it doesn't exist, then open in write mode
                # b paremeter specifies to read in binary mode (rather than in text encoding modes (e.g., utf))
                file.seek(0) # Moves stream position to the specified position 
                file.write(self.key) 
                file.truncate() # Truncates size of file at a specified position, defaults to current file stream position
                
        return self.key
    
    def importKey(self, path: Union[str, bytes, os.PathLike] = "data/fernetkey") -> bytes:
        """
        Imports a key and initializes Fernet cryptographer with it

        Args:
            path: Path to file that contains a Fernet key
            
        Returns: 
            bytes with imported key
        """
        with open(b"data/fernetkey", "r+b") as file: # Import key from file
            # r+ paramter specifies read and write
            # b paremeter specifies to read in binary mode (rather than in text encoding modes (e.g., utf))
            self.key = file.read()
        self.fernet = Fernet(self.key)
        return self.key
        
    def encryptFile(self, targ_file: Union[str, bytes, os.PathLike]):
        if self.fernet: # If Fernet has been instantiated by generating or importing key
            with open(targ_file, "r+b") as file: 
                file_original = file.read()
                file_encrypted = self.fernet.encrypt(file_original) # Encrypt file using encryptor
                file.seek(0) # Moves stream position to the specified position 
                file.write(file_encrypted) # Overwrite info with encrypted info 
                file.truncate() # Truncates size of file at a specified position, defaults to current file stream position
                # .truncate() documentation: https://www.w3schools.com/python/ref_file_truncate.asp#:~:text=The%20truncate()%20method%20resizes,current%20position%20will%20be%20used.
        else:
            print("No fernet instance detected")
        return self
    
    def decryptFile(self, targ_file: Union[str, bytes, os.PathLike]):
        if self.fernet: # If Fernet has been instantiated by generating or importing key
            with open(targ_file, "r+b") as file:
                file_encrypted = file.read()
                file_decrypted = self.fernet.decrypt(file_encrypted)
                file.seek(0) # Move stream position to beginning
                file.write(file_decrypted)
                file.truncate() # Truncate remaining data
        else:
            print("No fernet instance detected")
        return self
    

         