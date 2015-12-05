! 
! File:          bHYPRE_SStructVector_type.F90
! Symbol:        bHYPRE.SStructVector-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructVector
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructVector_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructVector.
! 

module bHYPRE_SStructVector_type
  use sidl
  type bHYPRE_SStructVector_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructVector_t

  type bHYPRE_SStructVector_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVector_1d

  type bHYPRE_SStructVector_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVector_2d

  type bHYPRE_SStructVector_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVector_3d

  type bHYPRE_SStructVector_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVector_4d

  type bHYPRE_SStructVector_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVector_5d

  type bHYPRE_SStructVector_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVector_6d

  type bHYPRE_SStructVector_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVector_7d

end module bHYPRE_SStructVector_type
