! 
! File:          bHYPRE_BiCGSTAB_type.F90
! Symbol:        bHYPRE.BiCGSTAB-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.BiCGSTAB
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_BiCGSTAB_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.BiCGSTAB.
! 

module bHYPRE_BiCGSTAB_type
  use sidl
  type bHYPRE_BiCGSTAB_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_BiCGSTAB_t

  type bHYPRE_BiCGSTAB_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BiCGSTAB_1d

  type bHYPRE_BiCGSTAB_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BiCGSTAB_2d

  type bHYPRE_BiCGSTAB_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BiCGSTAB_3d

  type bHYPRE_BiCGSTAB_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BiCGSTAB_4d

  type bHYPRE_BiCGSTAB_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BiCGSTAB_5d

  type bHYPRE_BiCGSTAB_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BiCGSTAB_6d

  type bHYPRE_BiCGSTAB_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_BiCGSTAB_7d

end module bHYPRE_BiCGSTAB_type
