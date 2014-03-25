! 
! File:          bHYPRE_SStructMatrixView_type.F90
! Symbol:        bHYPRE.SStructMatrixView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructMatrixView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructMatrixView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructMatrixView.
! 

module bHYPRE_SStructMatrixView_type
  use sidl
  type bHYPRE_SStructMatrixView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructMatrixView_t

  type bHYPRE_SStructMatrixView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixView_1d

  type bHYPRE_SStructMatrixView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixView_2d

  type bHYPRE_SStructMatrixView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixView_3d

  type bHYPRE_SStructMatrixView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixView_4d

  type bHYPRE_SStructMatrixView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixView_5d

  type bHYPRE_SStructMatrixView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixView_6d

  type bHYPRE_SStructMatrixView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixView_7d

end module bHYPRE_SStructMatrixView_type
