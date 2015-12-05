#
# File:  sidlBaseException.py
# Copyright (c) 2005 The Regents of the University of California
# $Revision: 1.5 $
# $Date: 2006/08/29 22:29:27 $
#

class sidlBaseException(Exception):
    """Base class for all SIDL Exception classes"""

    def __init__(self, exception):
        self.exception = exception
