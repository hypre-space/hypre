! 
! File:          bHYPRE_StructVector_type.F90
! Symbol:        bHYPRE.StructVector-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructVector
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructVector_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructVector.
! 

module bHYPRE_StructVector_type
  use sidl
  type bHYPRE_StructVector_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructVector_t

  type bHYPRE_StructVector_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVector_1d

  type bHYPRE_StructVector_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVector_2d

  type bHYPRE_StructVector_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVector_3d

  type bHYPRE_StructVector_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVector_4d

  type bHYPRE_StructVector_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVector_5d

  type bHYPRE_StructVector_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVector_6d

  type bHYPRE_StructVector_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVector_7d

end module bHYPRE_StructVector_type
