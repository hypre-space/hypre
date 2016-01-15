! 
! File:          bHYPRE_HGMRES_type.F90
! Symbol:        bHYPRE.HGMRES-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.HGMRES
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_HGMRES_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.HGMRES.
! 

module bHYPRE_HGMRES_type
  use sidl
  type bHYPRE_HGMRES_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_HGMRES_t

  type bHYPRE_HGMRES_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HGMRES_1d

  type bHYPRE_HGMRES_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HGMRES_2d

  type bHYPRE_HGMRES_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HGMRES_3d

  type bHYPRE_HGMRES_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HGMRES_4d

  type bHYPRE_HGMRES_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HGMRES_5d

  type bHYPRE_HGMRES_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HGMRES_6d

  type bHYPRE_HGMRES_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_HGMRES_7d

end module bHYPRE_HGMRES_type
