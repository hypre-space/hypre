! 
! File:          bHYPRE_StructMatrix_type.F90
! Symbol:        bHYPRE.StructMatrix-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructMatrix
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructMatrix_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructMatrix.
! 

module bHYPRE_StructMatrix_type
  use sidl
  type bHYPRE_StructMatrix_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructMatrix_t

  type bHYPRE_StructMatrix_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrix_1d

  type bHYPRE_StructMatrix_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrix_2d

  type bHYPRE_StructMatrix_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrix_3d

  type bHYPRE_StructMatrix_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrix_4d

  type bHYPRE_StructMatrix_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrix_5d

  type bHYPRE_StructMatrix_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrix_6d

  type bHYPRE_StructMatrix_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructMatrix_7d

end module bHYPRE_StructMatrix_type
