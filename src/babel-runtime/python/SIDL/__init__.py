#
# File:          __init__.py
# Symbol:        SIDL-v0.8.2
# Symbol Type:   package
# Babel Version: 0.8.4
# Release:       $Name: V1-9-0b $
# Revision:      @(#) $Id: __init__.py,v 1.4 2003/04/07 21:44:28 painter Exp $
# Description:   package initialization code
# 
# Copyright (c) 2000-2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the Components Team <components@llnl.gov>
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
# 
# WARNING: Automatically generated; changes will be lost
# 
# babel-version = 0.8.4
#


"""The <code>SIDL</code> package contains the fundamental type and interface
definitions for the <code>SIDL</code> interface definition language.  It
defines common run-time libraries and common base classes and interfaces.
Every interface implicitly inherits from <code>SIDL.BaseInterface</code>
and every class implicitly inherits from <code>SIDL.BaseClass</code>.

"""

all = [
   "BaseException",
   "BaseInterface",
   "BaseClass",
   "ClassInfo",
   "DLL",
   "ClassInfoI",
   "Loader" ]
