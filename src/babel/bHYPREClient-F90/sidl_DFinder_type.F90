! 
! File:          sidl_DFinder_type.F90
! Symbol:        sidl.DFinder-v0.9.15
! Symbol Type:   class
! Babel Version: 1.0.0
! Release:       $Name: V2-2-0b $
! Revision:      @(#) $Id: sidl_DFinder_type.F90,v 1.1 2007/02/06 01:23:06 painter Exp $
! Description:   Client-side module for sidl.DFinder
! 
! Copyright (c) 2000-2002, The Regents of the University of California.
! Produced at the Lawrence Livermore National Laboratory.
! Written by the Components Team <components@llnl.gov>
! All rights reserved.
! 
! This file is part of Babel. For more information, see
! http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
! for Our Notice and the LICENSE file for the GNU Lesser General Public
! License.
! 
! This program is free software; you can redistribute it and/or modify it
! under the terms of the GNU Lesser General Public License (as published by
! the Free Software Foundation) version 2.1 dated February 1999.
! 
! This program is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
! conditions of the GNU Lesser General Public License for more details.
! 
! You should have recieved a copy of the GNU Lesser General Public License
! along with this program; if not, write to the Free Software Foundation,
! Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "sidl_DFinder_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type sidl.DFinder.
! 

module sidl_DFinder_type
  use sidl
  type sidl_DFinder_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type sidl_DFinder_t

  type sidl_DFinder_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_DFinder_1d

  type sidl_DFinder_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_DFinder_2d

  type sidl_DFinder_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_DFinder_3d

  type sidl_DFinder_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_DFinder_4d

  type sidl_DFinder_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_DFinder_5d

  type sidl_DFinder_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_DFinder_6d

  type sidl_DFinder_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_DFinder_7d

end module sidl_DFinder_type
