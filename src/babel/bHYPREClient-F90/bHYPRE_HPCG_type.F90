! 
! File:          bHYPRE_HPCG_type.F90
! Symbol:        bHYPRE.HPCG-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.HPCG
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_HPCG_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.HPCG.
! 

module bHYPRE_HPCG_type
  use sidl
  type bHYPRE_HPCG_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_HPCG_t

  type bHYPRE_HPCG_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HPCG_1d

  type bHYPRE_HPCG_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HPCG_2d

  type bHYPRE_HPCG_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HPCG_3d

  type bHYPRE_HPCG_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HPCG_4d

  type bHYPRE_HPCG_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HPCG_5d

  type bHYPRE_HPCG_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HPCG_6d

  type bHYPRE_HPCG_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HPCG_7d

end module bHYPRE_HPCG_type
