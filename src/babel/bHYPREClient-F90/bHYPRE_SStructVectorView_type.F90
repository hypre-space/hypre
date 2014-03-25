! 
! File:          bHYPRE_SStructVectorView_type.F90
! Symbol:        bHYPRE.SStructVectorView-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructVectorView
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructVectorView_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructVectorView.
! 

module bHYPRE_SStructVectorView_type
  use sidl
  type bHYPRE_SStructVectorView_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructVectorView_t

  type bHYPRE_SStructVectorView_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVectorView_1d

  type bHYPRE_SStructVectorView_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVectorView_2d

  type bHYPRE_SStructVectorView_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVectorView_3d

  type bHYPRE_SStructVectorView_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVectorView_4d

  type bHYPRE_SStructVectorView_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVectorView_5d

  type bHYPRE_SStructVectorView_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVectorView_6d

  type bHYPRE_SStructVectorView_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructVectorView_7d

end module bHYPRE_SStructVectorView_type
