! 
! File:          bHYPRE_SStructMatrixVectorView_type.F90
! Symbol:        bHYPRE.SStructMatrixVectorView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructMatrixVectorView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructMatrixVectorView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructMatrixVectorView.
! 

module bHYPRE_SStructMatrixVectorView_type
  use sidl
  type bHYPRE_SStructMatrixVectorView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructMatrixVectorView_t

  type bHYPRE_SStructMatrixVectorView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixVectorView_1d

  type bHYPRE_SStructMatrixVectorView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixVectorView_2d

  type bHYPRE_SStructMatrixVectorView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixVectorView_3d

  type bHYPRE_SStructMatrixVectorView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixVectorView_4d

  type bHYPRE_SStructMatrixVectorView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixVectorView_5d

  type bHYPRE_SStructMatrixVectorView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixVectorView_6d

  type bHYPRE_SStructMatrixVectorView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrixVectorView_7d

end module bHYPRE_SStructMatrixVectorView_type
