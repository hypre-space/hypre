! 
! File:          bHYPRE_Vector_type.F90
! Symbol:        bHYPRE.Vector-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.Vector
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_Vector_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.Vector.
! 

module bHYPRE_Vector_type
  use sidl
  type bHYPRE_Vector_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_Vector_t

  type bHYPRE_Vector_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Vector_1d

  type bHYPRE_Vector_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Vector_2d

  type bHYPRE_Vector_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Vector_3d

  type bHYPRE_Vector_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Vector_4d

  type bHYPRE_Vector_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Vector_5d

  type bHYPRE_Vector_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Vector_6d

  type bHYPRE_Vector_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Vector_7d

end module bHYPRE_Vector_type
