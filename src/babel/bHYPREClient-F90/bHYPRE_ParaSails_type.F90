! 
! File:          bHYPRE_ParaSails_type.F90
! Symbol:        bHYPRE.ParaSails-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.ParaSails
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_ParaSails_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.ParaSails.
! 

module bHYPRE_ParaSails_type
  use sidl
  type bHYPRE_ParaSails_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_ParaSails_t

  type bHYPRE_ParaSails_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParaSails_1d

  type bHYPRE_ParaSails_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParaSails_2d

  type bHYPRE_ParaSails_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParaSails_3d

  type bHYPRE_ParaSails_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParaSails_4d

  type bHYPRE_ParaSails_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParaSails_5d

  type bHYPRE_ParaSails_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParaSails_6d

  type bHYPRE_ParaSails_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParaSails_7d

end module bHYPRE_ParaSails_type
