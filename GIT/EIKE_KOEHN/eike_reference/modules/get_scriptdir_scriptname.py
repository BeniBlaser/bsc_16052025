"""
Function collection regarding the scriptname and script directory.
author: Eike E. Koehn
date: Apr 6, 2023 
"""

import os

def get_scriptdir_scriptname(__file__):
    """
    author:      Eike E. Koehn
    date:        Apr 6, 2023
    description: get the directory and the name of the script
    """
    scriptdir = os.getcwd()+'/'
    scriptname = os.path.basename(__file__)
    return scriptdir, scriptname

    