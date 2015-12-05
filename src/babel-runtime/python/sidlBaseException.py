#
# File:  sidlBaseException.py
# Copyright (c) 2005 The Regents of the University of California
# $Revision: 1.6 $
# $Date: 2007/09/27 19:35:21 $
#

class sidlBaseException(Exception):
    """Base class for all SIDL Exception classes"""

    def __init__(self, exception):
        self.exception = exception
