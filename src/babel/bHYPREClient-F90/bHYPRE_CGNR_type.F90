! 
! File:          bHYPRE_CGNR_type.F90
! Symbol:        bHYPRE.CGNR-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.CGNR
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_CGNR_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.CGNR.
! 

module bHYPRE_CGNR_type
  use sidl
  type bHYPRE_CGNR_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_CGNR_t

  type bHYPRE_CGNR_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CGNR_1d

  type bHYPRE_CGNR_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CGNR_2d

  type bHYPRE_CGNR_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CGNR_3d

  type bHYPRE_CGNR_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CGNR_4d

  type bHYPRE_CGNR_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CGNR_5d

  type bHYPRE_CGNR_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CGNR_6d

  type bHYPRE_CGNR_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_CGNR_7d

end module bHYPRE_CGNR_type
