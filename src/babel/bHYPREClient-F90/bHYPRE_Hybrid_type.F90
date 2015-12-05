! 
! File:          bHYPRE_Hybrid_type.F90
! Symbol:        bHYPRE.Hybrid-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.Hybrid
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_Hybrid_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.Hybrid.
! 

module bHYPRE_Hybrid_type
  use sidl
  type bHYPRE_Hybrid_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_Hybrid_t

  type bHYPRE_Hybrid_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Hybrid_1d

  type bHYPRE_Hybrid_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Hybrid_2d

  type bHYPRE_Hybrid_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Hybrid_3d

  type bHYPRE_Hybrid_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Hybrid_4d

  type bHYPRE_Hybrid_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Hybrid_5d

  type bHYPRE_Hybrid_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Hybrid_6d

  type bHYPRE_Hybrid_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Hybrid_7d

end module bHYPRE_Hybrid_type
