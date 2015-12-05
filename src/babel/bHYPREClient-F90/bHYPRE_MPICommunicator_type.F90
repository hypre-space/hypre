! 
! File:          bHYPRE_MPICommunicator_type.F90
! Symbol:        bHYPRE.MPICommunicator-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.MPICommunicator
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_MPICommunicator_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.MPICommunicator.
! 

module bHYPRE_MPICommunicator_type
  use sidl
  type bHYPRE_MPICommunicator_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_MPICommunicator_t

  type bHYPRE_MPICommunicator_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MPICommunicator_1d

  type bHYPRE_MPICommunicator_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MPICommunicator_2d

  type bHYPRE_MPICommunicator_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MPICommunicator_3d

  type bHYPRE_MPICommunicator_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MPICommunicator_4d

  type bHYPRE_MPICommunicator_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MPICommunicator_5d

  type bHYPRE_MPICommunicator_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MPICommunicator_6d

  type bHYPRE_MPICommunicator_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_MPICommunicator_7d

end module bHYPRE_MPICommunicator_type
