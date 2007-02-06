! 
! File:          bHYPRE_SStructParCSRVector_type.F90
! Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructParCSRVector
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructParCSRVector_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructParCSRVector.
! 

module bHYPRE_SStructParCSRVector_type
  use sidl
  type bHYPRE_SStructParCSRVector_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructParCSRVector_t

  type bHYPRE_SStructParCSRVector_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRVector_1d

  type bHYPRE_SStructParCSRVector_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRVector_2d

  type bHYPRE_SStructParCSRVector_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRVector_3d

  type bHYPRE_SStructParCSRVector_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRVector_4d

  type bHYPRE_SStructParCSRVector_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRVector_5d

  type bHYPRE_SStructParCSRVector_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRVector_6d

  type bHYPRE_SStructParCSRVector_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRVector_7d

end module bHYPRE_SStructParCSRVector_type
