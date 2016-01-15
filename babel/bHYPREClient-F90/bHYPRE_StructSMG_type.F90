! 
! File:          bHYPRE_StructSMG_type.F90
! Symbol:        bHYPRE.StructSMG-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructSMG
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructSMG_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructSMG.
! 

module bHYPRE_StructSMG_type
  use sidl
  type bHYPRE_StructSMG_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructSMG_t

  type bHYPRE_StructSMG_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructSMG_1d

  type bHYPRE_StructSMG_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructSMG_2d

  type bHYPRE_StructSMG_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructSMG_3d

  type bHYPRE_StructSMG_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructSMG_4d

  type bHYPRE_StructSMG_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructSMG_5d

  type bHYPRE_StructSMG_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructSMG_6d

  type bHYPRE_StructSMG_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructSMG_7d

end module bHYPRE_StructSMG_type
