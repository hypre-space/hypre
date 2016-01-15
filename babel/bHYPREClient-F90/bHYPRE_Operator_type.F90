! 
! File:          bHYPRE_Operator_type.F90
! Symbol:        bHYPRE.Operator-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.Operator
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_Operator_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.Operator.
! 

module bHYPRE_Operator_type
  use sidl
  type bHYPRE_Operator_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_Operator_t

  type bHYPRE_Operator_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Operator_1d

  type bHYPRE_Operator_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Operator_2d

  type bHYPRE_Operator_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Operator_3d

  type bHYPRE_Operator_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Operator_4d

  type bHYPRE_Operator_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Operator_5d

  type bHYPRE_Operator_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Operator_6d

  type bHYPRE_Operator_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Operator_7d

end module bHYPRE_Operator_type
