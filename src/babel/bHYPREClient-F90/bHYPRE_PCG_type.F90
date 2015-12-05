! 
! File:          bHYPRE_PCG_type.F90
! Symbol:        bHYPRE.PCG-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.PCG
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_PCG_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.PCG.
! 

module bHYPRE_PCG_type
  use sidl
  type bHYPRE_PCG_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_PCG_t

  type bHYPRE_PCG_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PCG_1d

  type bHYPRE_PCG_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PCG_2d

  type bHYPRE_PCG_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PCG_3d

  type bHYPRE_PCG_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PCG_4d

  type bHYPRE_PCG_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PCG_5d

  type bHYPRE_PCG_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PCG_6d

  type bHYPRE_PCG_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PCG_7d

end module bHYPRE_PCG_type
