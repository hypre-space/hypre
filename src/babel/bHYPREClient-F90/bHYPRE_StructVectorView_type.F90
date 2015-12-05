! 
! File:          bHYPRE_StructVectorView_type.F90
! Symbol:        bHYPRE.StructVectorView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructVectorView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructVectorView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructVectorView.
! 

module bHYPRE_StructVectorView_type
  use sidl
  type bHYPRE_StructVectorView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructVectorView_t

  type bHYPRE_StructVectorView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVectorView_1d

  type bHYPRE_StructVectorView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVectorView_2d

  type bHYPRE_StructVectorView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVectorView_3d

  type bHYPRE_StructVectorView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVectorView_4d

  type bHYPRE_StructVectorView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVectorView_5d

  type bHYPRE_StructVectorView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVectorView_6d

  type bHYPRE_StructVectorView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructVectorView_7d

end module bHYPRE_StructVectorView_type
