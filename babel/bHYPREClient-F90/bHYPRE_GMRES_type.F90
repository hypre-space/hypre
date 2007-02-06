! 
! File:          bHYPRE_GMRES_type.F90
! Symbol:        bHYPRE.GMRES-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.GMRES
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_GMRES_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.GMRES.
! 

module bHYPRE_GMRES_type
  use sidl
  type bHYPRE_GMRES_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_GMRES_t

  type bHYPRE_GMRES_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_GMRES_1d

  type bHYPRE_GMRES_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_GMRES_2d

  type bHYPRE_GMRES_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_GMRES_3d

  type bHYPRE_GMRES_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_GMRES_4d

  type bHYPRE_GMRES_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_GMRES_5d

  type bHYPRE_GMRES_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_GMRES_6d

  type bHYPRE_GMRES_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_GMRES_7d

end module bHYPRE_GMRES_type
