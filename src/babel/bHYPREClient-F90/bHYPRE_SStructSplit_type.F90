! 
! File:          bHYPRE_SStructSplit_type.F90
! Symbol:        bHYPRE.SStructSplit-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructSplit
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructSplit_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructSplit.
! 

module bHYPRE_SStructSplit_type
  use sidl
  type bHYPRE_SStructSplit_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructSplit_t

  type bHYPRE_SStructSplit_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructSplit_1d

  type bHYPRE_SStructSplit_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructSplit_2d

  type bHYPRE_SStructSplit_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructSplit_3d

  type bHYPRE_SStructSplit_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructSplit_4d

  type bHYPRE_SStructSplit_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructSplit_5d

  type bHYPRE_SStructSplit_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructSplit_6d

  type bHYPRE_SStructSplit_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructSplit_7d

end module bHYPRE_SStructSplit_type
