! 
! File:          bHYPRE_SStructMatrix_type.F90
! Symbol:        bHYPRE.SStructMatrix-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructMatrix
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructMatrix_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructMatrix.
! 

module bHYPRE_SStructMatrix_type
  use sidl
  type bHYPRE_SStructMatrix_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructMatrix_t

  type bHYPRE_SStructMatrix_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrix_1d

  type bHYPRE_SStructMatrix_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrix_2d

  type bHYPRE_SStructMatrix_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrix_3d

  type bHYPRE_SStructMatrix_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrix_4d

  type bHYPRE_SStructMatrix_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrix_5d

  type bHYPRE_SStructMatrix_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrix_6d

  type bHYPRE_SStructMatrix_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructMatrix_7d

end module bHYPRE_SStructMatrix_type
