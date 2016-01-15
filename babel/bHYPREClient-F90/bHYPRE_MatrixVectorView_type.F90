! 
! File:          bHYPRE_MatrixVectorView_type.F90
! Symbol:        bHYPRE.MatrixVectorView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.MatrixVectorView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_MatrixVectorView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.MatrixVectorView.
! 

module bHYPRE_MatrixVectorView_type
  use sidl
  type bHYPRE_MatrixVectorView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_MatrixVectorView_t

  type bHYPRE_MatrixVectorView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MatrixVectorView_1d

  type bHYPRE_MatrixVectorView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MatrixVectorView_2d

  type bHYPRE_MatrixVectorView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MatrixVectorView_3d

  type bHYPRE_MatrixVectorView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MatrixVectorView_4d

  type bHYPRE_MatrixVectorView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MatrixVectorView_5d

  type bHYPRE_MatrixVectorView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MatrixVectorView_6d

  type bHYPRE_MatrixVectorView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MatrixVectorView_7d

end module bHYPRE_MatrixVectorView_type
