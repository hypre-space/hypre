#
# File:  sidlBaseException.py
# Copyright (c) 2005 The Regents of the University of California
# $Revision$
# $Date$
#

class sidlBaseException(Exception):
    """Base class for all SIDL Exception classes"""

    def __init__(self, exception):
        self.exception = exception
