! 
! File:          bHYPRE_SStructStencil_type.F90
! Symbol:        bHYPRE.SStructStencil-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructStencil
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructStencil_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructStencil.
! 

module bHYPRE_SStructStencil_type
  use sidl
  type bHYPRE_SStructStencil_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructStencil_t

  type bHYPRE_SStructStencil_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructStencil_1d

  type bHYPRE_SStructStencil_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructStencil_2d

  type bHYPRE_SStructStencil_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructStencil_3d

  type bHYPRE_SStructStencil_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructStencil_4d

  type bHYPRE_SStructStencil_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructStencil_5d

  type bHYPRE_SStructStencil_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructStencil_6d

  type bHYPRE_SStructStencil_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructStencil_7d

end module bHYPRE_SStructStencil_type
