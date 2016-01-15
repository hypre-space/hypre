! 
! File:          bHYPRE_BoomerAMG_type.F90
! Symbol:        bHYPRE.BoomerAMG-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.BoomerAMG
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_BoomerAMG_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.BoomerAMG.
! 

module bHYPRE_BoomerAMG_type
  use sidl
  type bHYPRE_BoomerAMG_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_BoomerAMG_t

  type bHYPRE_BoomerAMG_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BoomerAMG_1d

  type bHYPRE_BoomerAMG_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BoomerAMG_2d

  type bHYPRE_BoomerAMG_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BoomerAMG_3d

  type bHYPRE_BoomerAMG_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BoomerAMG_4d

  type bHYPRE_BoomerAMG_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BoomerAMG_5d

  type bHYPRE_BoomerAMG_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BoomerAMG_6d

  type bHYPRE_BoomerAMG_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BoomerAMG_7d

end module bHYPRE_BoomerAMG_type
