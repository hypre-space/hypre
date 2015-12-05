! 
! File:          bHYPRE_StructStencil_type.F90
! Symbol:        bHYPRE.StructStencil-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructStencil
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructStencil_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructStencil.
! 

module bHYPRE_StructStencil_type
  use sidl
  type bHYPRE_StructStencil_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructStencil_t

  type bHYPRE_StructStencil_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructStencil_1d

  type bHYPRE_StructStencil_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructStencil_2d

  type bHYPRE_StructStencil_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructStencil_3d

  type bHYPRE_StructStencil_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructStencil_4d

  type bHYPRE_StructStencil_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructStencil_5d

  type bHYPRE_StructStencil_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructStencil_6d

  type bHYPRE_StructStencil_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructStencil_7d

end module bHYPRE_StructStencil_type
