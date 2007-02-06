! 
! File:          bHYPRE_IdentitySolver_type.F90
! Symbol:        bHYPRE.IdentitySolver-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.IdentitySolver
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_IdentitySolver_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.IdentitySolver.
! 

module bHYPRE_IdentitySolver_type
  use sidl
  type bHYPRE_IdentitySolver_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_IdentitySolver_t

  type bHYPRE_IdentitySolver_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IdentitySolver_1d

  type bHYPRE_IdentitySolver_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IdentitySolver_2d

  type bHYPRE_IdentitySolver_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IdentitySolver_3d

  type bHYPRE_IdentitySolver_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IdentitySolver_4d

  type bHYPRE_IdentitySolver_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IdentitySolver_5d

  type bHYPRE_IdentitySolver_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IdentitySolver_6d

  type bHYPRE_IdentitySolver_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IdentitySolver_7d

end module bHYPRE_IdentitySolver_type
