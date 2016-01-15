! 
! File:          bHYPRE_IJParCSRVector_type.F90
! Symbol:        bHYPRE.IJParCSRVector-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.IJParCSRVector
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_IJParCSRVector_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.IJParCSRVector.
! 

module bHYPRE_IJParCSRVector_type
  use sidl
  type bHYPRE_IJParCSRVector_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_IJParCSRVector_t

  type bHYPRE_IJParCSRVector_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRVector_1d

  type bHYPRE_IJParCSRVector_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRVector_2d

  type bHYPRE_IJParCSRVector_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRVector_3d

  type bHYPRE_IJParCSRVector_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRVector_4d

  type bHYPRE_IJParCSRVector_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRVector_5d

  type bHYPRE_IJParCSRVector_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRVector_6d

  type bHYPRE_IJParCSRVector_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRVector_7d

end module bHYPRE_IJParCSRVector_type
