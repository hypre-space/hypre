! 
! File:          bHYPRE_SStructDiagScale_type.F90
! Symbol:        bHYPRE.SStructDiagScale-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructDiagScale
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructDiagScale_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructDiagScale.
! 

module bHYPRE_SStructDiagScale_type
  use sidl
  type bHYPRE_SStructDiagScale_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructDiagScale_t

  type bHYPRE_SStructDiagScale_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructDiagScale_1d

  type bHYPRE_SStructDiagScale_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructDiagScale_2d

  type bHYPRE_SStructDiagScale_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructDiagScale_3d

  type bHYPRE_SStructDiagScale_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructDiagScale_4d

  type bHYPRE_SStructDiagScale_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructDiagScale_5d

  type bHYPRE_SStructDiagScale_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructDiagScale_6d

  type bHYPRE_SStructDiagScale_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructDiagScale_7d

end module bHYPRE_SStructDiagScale_type
