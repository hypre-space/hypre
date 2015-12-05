! 
! File:          bHYPRE_CoefficientAccess_type.F90
! Symbol:        bHYPRE.CoefficientAccess-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.CoefficientAccess
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_CoefficientAccess_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.CoefficientAccess.
! 

module bHYPRE_CoefficientAccess_type
  use sidl
  type bHYPRE_CoefficientAccess_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_CoefficientAccess_t

  type bHYPRE_CoefficientAccess_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CoefficientAccess_1d

  type bHYPRE_CoefficientAccess_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CoefficientAccess_2d

  type bHYPRE_CoefficientAccess_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CoefficientAccess_3d

  type bHYPRE_CoefficientAccess_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CoefficientAccess_4d

  type bHYPRE_CoefficientAccess_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CoefficientAccess_5d

  type bHYPRE_CoefficientAccess_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CoefficientAccess_6d

  type bHYPRE_CoefficientAccess_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CoefficientAccess_7d

end module bHYPRE_CoefficientAccess_type
