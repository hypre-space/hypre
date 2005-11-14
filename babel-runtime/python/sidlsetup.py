#! /usr/bin/env python
# Python setup.py to build sidl python support libraries
#
# Copyright (c) 2000-2003, The Regents of the University of Calfornia.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the Components Team <components@llnl.gov>
# UCRL-CODE-2002-054
# All rights reserved.
# 
# This file is part of Babel. For more information, see
# http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
# for Our Notice and the LICENSE file for the GNU Lesser General Public
# License.
# 
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
# 
# You should have recieved a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
from distutils.core import setup,Extension
from re import compile
import sys

inc_re = compile('^--include-dirs=(.*)$')
lib_re = compile('^--library-dirs=(.*)$')
exlib_re = compile('^--extra-library=(.*)$')
old_argv = sys.argv
sys.argv = []
inc_dirs = ['.']
lib_dirs = []
libs=['sidl']

for i in old_argv:
  m = inc_re.match(i)
  if (m):
    if (len(m.group(1))): inc_dirs.append(m.group(1))
  else:
    m = lib_re.match(i)
    if (m):
      if (len(m.group(1))): lib_dirs.append(m.group(1))
    else:
      m = exlib_re.match(i)
      if (m):
        if (len(m.group(1))): libs.append(m.group(1))
      else:
        sys.argv.append(i)
      
setup(name="babel",
      author="Tom Epperly",
      version="0.10.12",
      description="Build Python support extension modules for sidl",
      author_email="components@llnl.gov",
      url="http://www.llnl.gov/CASC/components/",
      include_dirs=inc_dirs,
      headers = ["sidlObjA.h", "sidlPyArrays.h"],
      py_modules = [ 'sidlBaseException' ],
      ext_modules = [
    Extension('sidlObjA',
              ["sidlObjA.c"],
              library_dirs=lib_dirs,
              libraries=libs),
    Extension('sidlPyArrays',
              ["sidlPyArrays.c"],
              library_dirs=lib_dirs,
              libraries=libs)])
