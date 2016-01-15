! 
! File:          bHYPRE_SStructParCSRMatrix_type.F90
! Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructParCSRMatrix
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructParCSRMatrix_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructParCSRMatrix.
! 

module bHYPRE_SStructParCSRMatrix_type
  use sidl
  type bHYPRE_SStructParCSRMatrix_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructParCSRMatrix_t

  type bHYPRE_SStructParCSRMatrix_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRMatrix_1d

  type bHYPRE_SStructParCSRMatrix_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRMatrix_2d

  type bHYPRE_SStructParCSRMatrix_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRMatrix_3d

  type bHYPRE_SStructParCSRMatrix_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRMatrix_4d

  type bHYPRE_SStructParCSRMatrix_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRMatrix_5d

  type bHYPRE_SStructParCSRMatrix_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRMatrix_6d

  type bHYPRE_SStructParCSRMatrix_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructParCSRMatrix_7d

end module bHYPRE_SStructParCSRMatrix_type
