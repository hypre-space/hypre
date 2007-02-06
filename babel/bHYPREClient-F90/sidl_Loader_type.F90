! 
! File:          sidl_Loader_type.F90
! Symbol:        sidl.Loader-v0.9.15
! Symbol Type:   class
! Babel Version: 1.0.0
! Release:       $Name$
! Revision:      @(#) $Id$
! Description:   Client-side module for sidl.Loader
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

#include "sidl_Loader_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type sidl.Loader.
! 

module sidl_Loader_type
  use sidl
  type sidl_Loader_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type sidl_Loader_t

  type sidl_Loader_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_Loader_1d

  type sidl_Loader_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_Loader_2d

  type sidl_Loader_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_Loader_3d

  type sidl_Loader_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_Loader_4d

  type sidl_Loader_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_Loader_5d

  type sidl_Loader_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_Loader_6d

  type sidl_Loader_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl_Loader_7d

end module sidl_Loader_type
