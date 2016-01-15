! 
! File:          bHYPRE_IJMatrixView_type.F90
! Symbol:        bHYPRE.IJMatrixView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.IJMatrixView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_IJMatrixView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.IJMatrixView.
! 

module bHYPRE_IJMatrixView_type
  use sidl
  type bHYPRE_IJMatrixView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_IJMatrixView_t

  type bHYPRE_IJMatrixView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJMatrixView_1d

  type bHYPRE_IJMatrixView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJMatrixView_2d

  type bHYPRE_IJMatrixView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJMatrixView_3d

  type bHYPRE_IJMatrixView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJMatrixView_4d

  type bHYPRE_IJMatrixView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJMatrixView_5d

  type bHYPRE_IJMatrixView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJMatrixView_6d

  type bHYPRE_IJMatrixView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJMatrixView_7d

end module bHYPRE_IJMatrixView_type
