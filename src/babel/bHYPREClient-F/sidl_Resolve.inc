C       
C       File:          sidl_Resolve.inc
C       Symbol:        sidl.Resolve-v0.9.15
C       Symbol Type:   enumeration
C       Babel Version: 1.0.4
C       Release:       $Name$
C       Revision:      @(#) $Id$
C       Description:   Client-side glue code for sidl.Resolve
C       
C       Copyright (c) 2000-2002, The Regents of the University of California.
C       Produced at the Lawrence Livermore National Laboratory.
C       Written by the Components Team <components@llnl.gov>
C       All rights reserved.
C       
C       This file is part of Babel. For more information, see
C       http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
C       for Our Notice and the LICENSE file for the GNU Lesser General Public
C       License.
C       
C       This program is free software; you can redistribute it and/or modify it
C       under the terms of the GNU Lesser General Public License (as published by
C       the Free Software Foundation) version 2.1 dated February 1999.
C       
C       This program is distributed in the hope that it will be useful, but
C       WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
C       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
C       conditions of the GNU Lesser General Public License for more details.
C       
C       You should have recieved a copy of the GNU Lesser General Public License
C       along with this program; if not, write to the Free Software Foundation,
C       Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
C       
C       WARNING: Automatically generated; changes will be lost
C       


C       
C       Symbol "sidl.Resolve" (version 0.9.15)
C       
C       When loading a dynmaically linked library, there are three
C       settings: LAZY, NOW, SCLRESOLVE
C       


        integer*4 LAZY
        parameter (LAZY = 0)
        integer*4 NOW
        parameter (NOW = 1)
        integer*4 SCLRESOLVE
        parameter (SCLRESOLVE = 2)
