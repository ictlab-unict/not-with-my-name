'''
    @Author: Roberto Leotta
    @Version: 1.0
    @Date: 08/16/2023

    Utils snippets.
'''
########## IMPORTs START ##########

# system imports
import os
import sys

# processing
import cv2
import numpy as np

###### IMPORTs - END ##############

###
###
###

###### CONSTANTs - START ##########

ROOT_DIR = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))
ALLOWED_FILES_EXT = ['jpg', 'png']

###### CONSTANTs - END ############

###
###
###

###### FUNCTIONs - START ##########

def p_info(message, n_tabs=0, end='\n', pre=''):
    """
    Function to print an info message.
    :param message: message to print in stdout
    :param n_tabs: initial blank tab(s). Default n_tabs=0
    :param end: end character for each printed message. Default end='\n'
    :param pre: prefix to prepend to the message. Default pre=''
    """
    if message == '':
        print('**')
    else:
        tabs_str = ''
        for tab in range(0, n_tabs):
            tabs_str += '\t'
        print('{}** (I) ** {}{}'.format(pre, tabs_str, message), end=end)


def p_warning(message, n_tabs=0, end='\n', pre=''):
    """
    Function to print an warning message.
    :param message: message to print in stdout
    :param n_tabs: initial blank tab(s). Default n_tabs=0
    :param end: end character for each printed message. Default end='\n'
    :param pre: prefix to prepend to the message. Default pre=''
    """
    tabs_str = ''
    for tab in range(0, n_tabs):
        tabs_str += '\t'
    print('{}** (W) ** {}{}'.format(pre, tabs_str, message), end=end)


def p_error_n_exit(message, n_tabs=0):
    """
    Function to print an error message and exit.
    :param message: message to print in stdout
    :param n_tabs: initial blank tab(s). Default n_tabs=0
    """
    tabs_str = ''
    for tab in range(0, n_tabs):
        tabs_str += '\t'
    print('** (E) **{}- {}'.format(tabs_str, message))

    sys.exit(1)


def p_error(message, n_tabs=0):
    """
    Function to print an error message, without exit.
    :param message: message to print in stdout
    :param n_tabs: initial blank tab(s). Default n_tabs=0
    """
    if message == '':
        print('**')
    else:
        tabs_str = ''
        for tab in range(0, n_tabs):
            tabs_str += '\t'
        print('** (E) ** {}{}'.format(tabs_str, message))


def str2bool(v):
    """
    Convert string to boolean.
    :param v: string to convert in boolean
    :return bool: string converted in True or False
    """
    return v.lower() in ("yes", "y", "true", "t", "1")


def allowedFile(file_path):
    """
    Function to allow the right files.
    :param file_path: file_path
    :return bool: True if allowed, False otherwise
    """
    return True if file_path.split('.')[-1] in ALLOWED_FILES_EXT else False


def show(imgs, name='Results'):
    """
    Function to show image, or multiple images (vertical stucked)
    :param imgs: image or images as list. i.e.: ((img)) or ((img1, img2))
    :param name: name of the window to show
    """
    if imgs.__class__ != tuple:
        res = imgs
    else:
        res = np.vstack((imgs))

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, res.shape[1], res.shape[0])
    cv2.imshow(name, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###### FUNCTIONs - END ############

###
###
###

###### COMMENTs ###################