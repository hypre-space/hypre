! 
! File:          bHYPRE_SStructGrid_type.F90
! Symbol:        bHYPRE.SStructGrid-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructGrid
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructGrid_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructGrid.
! 

module bHYPRE_SStructGrid_type
  use sidl
  type bHYPRE_SStructGrid_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructGrid_t

  type bHYPRE_SStructGrid_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGrid_1d

  type bHYPRE_SStructGrid_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGrid_2d

  type bHYPRE_SStructGrid_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGrid_3d

  type bHYPRE_SStructGrid_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGrid_4d

  type bHYPRE_SStructGrid_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGrid_5d

  type bHYPRE_SStructGrid_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGrid_6d

  type bHYPRE_SStructGrid_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGrid_7d

end module bHYPRE_SStructGrid_type
