! 
! File:          bHYPRE_StructJacobi_type.F90
! Symbol:        bHYPRE.StructJacobi-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.StructJacobi
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_StructJacobi_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.StructJacobi.
! 

module bHYPRE_StructJacobi_type
  use sidl
  type bHYPRE_StructJacobi_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_StructJacobi_t

  type bHYPRE_StructJacobi_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructJacobi_1d

  type bHYPRE_StructJacobi_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructJacobi_2d

  type bHYPRE_StructJacobi_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructJacobi_3d

  type bHYPRE_StructJacobi_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructJacobi_4d

  type bHYPRE_StructJacobi_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructJacobi_5d

  type bHYPRE_StructJacobi_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructJacobi_6d

  type bHYPRE_StructJacobi_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_StructJacobi_7d

end module bHYPRE_StructJacobi_type
