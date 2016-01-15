! 
! File:          bHYPRE_ErrorHandler_type.F90
! Symbol:        bHYPRE.ErrorHandler-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.ErrorHandler
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_ErrorHandler_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.ErrorHandler.
! 

module bHYPRE_ErrorHandler_type
  use sidl
  type bHYPRE_ErrorHandler_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_ErrorHandler_t

  type bHYPRE_ErrorHandler_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ErrorHandler_1d

  type bHYPRE_ErrorHandler_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ErrorHandler_2d

  type bHYPRE_ErrorHandler_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ErrorHandler_3d

  type bHYPRE_ErrorHandler_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ErrorHandler_4d

  type bHYPRE_ErrorHandler_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ErrorHandler_5d

  type bHYPRE_ErrorHandler_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ErrorHandler_6d

  type bHYPRE_ErrorHandler_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ErrorHandler_7d

end module bHYPRE_ErrorHandler_type
