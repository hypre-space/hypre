! 
! File:          bHYPRE_IJParCSRMatrix_type.F90
! Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.IJParCSRMatrix
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_IJParCSRMatrix_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.IJParCSRMatrix.
! 

module bHYPRE_IJParCSRMatrix_type
  use sidl
  type bHYPRE_IJParCSRMatrix_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_IJParCSRMatrix_t

  type bHYPRE_IJParCSRMatrix_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRMatrix_1d

  type bHYPRE_IJParCSRMatrix_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRMatrix_2d

  type bHYPRE_IJParCSRMatrix_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRMatrix_3d

  type bHYPRE_IJParCSRMatrix_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRMatrix_4d

  type bHYPRE_IJParCSRMatrix_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRMatrix_5d

  type bHYPRE_IJParCSRMatrix_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRMatrix_6d

  type bHYPRE_IJParCSRMatrix_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJParCSRMatrix_7d

end module bHYPRE_IJParCSRMatrix_type
