! 
! File:          bHYPRE_Schwarz_type.F90
! Symbol:        bHYPRE.Schwarz-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.Schwarz
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_Schwarz_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.Schwarz.
! 

module bHYPRE_Schwarz_type
  use sidl
  type bHYPRE_Schwarz_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_Schwarz_t

  type bHYPRE_Schwarz_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Schwarz_1d

  type bHYPRE_Schwarz_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Schwarz_2d

  type bHYPRE_Schwarz_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Schwarz_3d

  type bHYPRE_Schwarz_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Schwarz_4d

  type bHYPRE_Schwarz_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Schwarz_5d

  type bHYPRE_Schwarz_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Schwarz_6d

  type bHYPRE_Schwarz_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Schwarz_7d

end module bHYPRE_Schwarz_type
