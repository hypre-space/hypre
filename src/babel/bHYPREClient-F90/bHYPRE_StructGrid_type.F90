! 
! File:          bHYPRE_StructGrid_type.F90
! Symbol:        bHYPRE.StructGrid-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructGrid
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructGrid_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructGrid.
! 

module bHYPRE_StructGrid_type
  use sidl
  type bHYPRE_StructGrid_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructGrid_t

  type bHYPRE_StructGrid_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructGrid_1d

  type bHYPRE_StructGrid_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructGrid_2d

  type bHYPRE_StructGrid_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructGrid_3d

  type bHYPRE_StructGrid_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructGrid_4d

  type bHYPRE_StructGrid_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructGrid_5d

  type bHYPRE_StructGrid_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructGrid_6d

  type bHYPRE_StructGrid_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructGrid_7d

end module bHYPRE_StructGrid_type
