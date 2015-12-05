! 
! File:          bHYPRE_StructDiagScale_type.F90
! Symbol:        bHYPRE.StructDiagScale-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructDiagScale
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructDiagScale_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructDiagScale.
! 

module bHYPRE_StructDiagScale_type
  use sidl
  type bHYPRE_StructDiagScale_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructDiagScale_t

  type bHYPRE_StructDiagScale_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructDiagScale_1d

  type bHYPRE_StructDiagScale_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructDiagScale_2d

  type bHYPRE_StructDiagScale_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructDiagScale_3d

  type bHYPRE_StructDiagScale_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructDiagScale_4d

  type bHYPRE_StructDiagScale_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructDiagScale_5d

  type bHYPRE_StructDiagScale_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructDiagScale_6d

  type bHYPRE_StructDiagScale_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructDiagScale_7d

end module bHYPRE_StructDiagScale_type
