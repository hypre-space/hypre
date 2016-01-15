! 
! File:          bHYPRE_Euclid_type.F90
! Symbol:        bHYPRE.Euclid-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.Euclid
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_Euclid_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.Euclid.
! 

module bHYPRE_Euclid_type
  use sidl
  type bHYPRE_Euclid_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_Euclid_t

  type bHYPRE_Euclid_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Euclid_1d

  type bHYPRE_Euclid_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Euclid_2d

  type bHYPRE_Euclid_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Euclid_3d

  type bHYPRE_Euclid_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Euclid_4d

  type bHYPRE_Euclid_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Euclid_5d

  type bHYPRE_Euclid_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Euclid_6d

  type bHYPRE_Euclid_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Euclid_7d

end module bHYPRE_Euclid_type
