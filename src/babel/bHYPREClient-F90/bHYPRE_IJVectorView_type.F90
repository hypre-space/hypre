! 
! File:          bHYPRE_IJVectorView_type.F90
! Symbol:        bHYPRE.IJVectorView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.IJVectorView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_IJVectorView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.IJVectorView.
! 

module bHYPRE_IJVectorView_type
  use sidl
  type bHYPRE_IJVectorView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_IJVectorView_t

  type bHYPRE_IJVectorView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJVectorView_1d

  type bHYPRE_IJVectorView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJVectorView_2d

  type bHYPRE_IJVectorView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJVectorView_3d

  type bHYPRE_IJVectorView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJVectorView_4d

  type bHYPRE_IJVectorView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJVectorView_5d

  type bHYPRE_IJVectorView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJVectorView_6d

  type bHYPRE_IJVectorView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_IJVectorView_7d

end module bHYPRE_IJVectorView_type
