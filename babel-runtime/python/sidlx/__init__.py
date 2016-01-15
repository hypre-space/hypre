#
# File:          __init__.py
# Symbol:        sidlx-v0.1
# Symbol Type:   package
# Babel Version: 1.0.4
# Description:   package initialization code
# 
# WARNING: Automatically generated; changes will be lost
# 
#


"""This package contains experimental extensions to sidl
and should not be used unless willing to tolerate its
disappearance in a following release
"""

__all__ = [
   "rmi" ]

try:
  from pkgutil import extend_path
  __path__ = extend_path(__path__, __name__)
except: # ignore all exceptions
  pass
