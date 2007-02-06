! 
! File:          bHYPRE_StructMatrixView_type.F90
! Symbol:        bHYPRE.StructMatrixView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructMatrixView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructMatrixView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructMatrixView.
! 

module bHYPRE_StructMatrixView_type
  use sidl
  type bHYPRE_StructMatrixView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructMatrixView_t

  type bHYPRE_StructMatrixView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrixView_1d

  type bHYPRE_StructMatrixView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrixView_2d

  type bHYPRE_StructMatrixView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrixView_3d

  type bHYPRE_StructMatrixView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrixView_4d

  type bHYPRE_StructMatrixView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrixView_5d

  type bHYPRE_StructMatrixView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrixView_6d

  type bHYPRE_StructMatrixView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrixView_7d

end module bHYPRE_StructMatrixView_type
