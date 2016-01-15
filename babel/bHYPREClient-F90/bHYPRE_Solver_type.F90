! 
! File:          bHYPRE_Solver_type.F90
! Symbol:        bHYPRE.Solver-v1.0.0
! Symbol Type:   interface
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.Solver
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_Solver_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.Solver.
! 

module bHYPRE_Solver_type
  use sidl
  type bHYPRE_Solver_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_Solver_t

  type bHYPRE_Solver_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Solver_1d

  type bHYPRE_Solver_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Solver_2d

  type bHYPRE_Solver_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Solver_3d

  type bHYPRE_Solver_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Solver_4d

  type bHYPRE_Solver_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Solver_5d

  type bHYPRE_Solver_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Solver_6d

  type bHYPRE_Solver_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_Solver_7d

end module bHYPRE_Solver_type
