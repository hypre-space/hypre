! 
! File:          bHYPRE_PreconditionedSolver_type.F90
! Symbol:        bHYPRE.PreconditionedSolver-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.PreconditionedSolver
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_PreconditionedSolver_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.PreconditionedSolver.
! 

module bHYPRE_PreconditionedSolver_type
  use sidl
  type bHYPRE_PreconditionedSolver_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_PreconditionedSolver_t

  type bHYPRE_PreconditionedSolver_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PreconditionedSolver_1d

  type bHYPRE_PreconditionedSolver_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PreconditionedSolver_2d

  type bHYPRE_PreconditionedSolver_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PreconditionedSolver_3d

  type bHYPRE_PreconditionedSolver_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PreconditionedSolver_4d

  type bHYPRE_PreconditionedSolver_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PreconditionedSolver_5d

  type bHYPRE_PreconditionedSolver_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PreconditionedSolver_6d

  type bHYPRE_PreconditionedSolver_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_PreconditionedSolver_7d

end module bHYPRE_PreconditionedSolver_type
