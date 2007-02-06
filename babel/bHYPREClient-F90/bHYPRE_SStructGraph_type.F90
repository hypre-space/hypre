! 
! File:          bHYPRE_SStructGraph_type.F90
! Symbol:        bHYPRE.SStructGraph-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.SStructGraph
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_SStructGraph_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.SStructGraph.
! 

module bHYPRE_SStructGraph_type
  use sidl
  type bHYPRE_SStructGraph_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_SStructGraph_t

  type bHYPRE_SStructGraph_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGraph_1d

  type bHYPRE_SStructGraph_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGraph_2d

  type bHYPRE_SStructGraph_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGraph_3d

  type bHYPRE_SStructGraph_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGraph_4d

  type bHYPRE_SStructGraph_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGraph_5d

  type bHYPRE_SStructGraph_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGraph_6d

  type bHYPRE_SStructGraph_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_SStructGraph_7d

end module bHYPRE_SStructGraph_type
