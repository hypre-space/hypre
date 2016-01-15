! 
! File:          bHYPRE_Pilut_type.F90
! Symbol:        bHYPRE.Pilut-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.Pilut
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_Pilut_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.Pilut.
! 

module bHYPRE_Pilut_type
  use sidl
  type bHYPRE_Pilut_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_Pilut_t

  type bHYPRE_Pilut_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Pilut_1d

  type bHYPRE_Pilut_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Pilut_2d

  type bHYPRE_Pilut_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Pilut_3d

  type bHYPRE_Pilut_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Pilut_4d

  type bHYPRE_Pilut_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Pilut_5d

  type bHYPRE_Pilut_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Pilut_6d

  type bHYPRE_Pilut_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Pilut_7d

end module bHYPRE_Pilut_type
