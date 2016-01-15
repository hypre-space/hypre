! 
! File:          bHYPRE_ProblemDefinition_type.F90
! Symbol:        bHYPRE.ProblemDefinition-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.ProblemDefinition
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_ProblemDefinition_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.ProblemDefinition.
! 

module bHYPRE_ProblemDefinition_type
  use sidl
  type bHYPRE_ProblemDefinition_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_ProblemDefinition_t

  type bHYPRE_ProblemDefinition_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ProblemDefinition_1d

  type bHYPRE_ProblemDefinition_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ProblemDefinition_2d

  type bHYPRE_ProblemDefinition_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ProblemDefinition_3d

  type bHYPRE_ProblemDefinition_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ProblemDefinition_4d

  type bHYPRE_ProblemDefinition_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ProblemDefinition_5d

  type bHYPRE_ProblemDefinition_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ProblemDefinition_6d

  type bHYPRE_ProblemDefinition_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ProblemDefinition_7d

end module bHYPRE_ProblemDefinition_type
