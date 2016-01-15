! 
! File:          bHYPRE_ParCSRDiagScale_type.F90
! Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.ParCSRDiagScale
! 
! WARNING: Automatically generated; changes will be lost
! 
! 

#include "bHYPRE_ParCSRDiagScale_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type bHYPRE.ParCSRDiagScale.
! 

module bHYPRE_ParCSRDiagScale_type
  use sidl
  type bHYPRE_ParCSRDiagScale_t
    sequence
    integer (kind=sidl_iorptr) :: d_ior
  end type bHYPRE_ParCSRDiagScale_t

  type bHYPRE_ParCSRDiagScale_1d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParCSRDiagScale_1d

  type bHYPRE_ParCSRDiagScale_2d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParCSRDiagScale_2d

  type bHYPRE_ParCSRDiagScale_3d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParCSRDiagScale_3d

  type bHYPRE_ParCSRDiagScale_4d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParCSRDiagScale_4d

  type bHYPRE_ParCSRDiagScale_5d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParCSRDiagScale_5d

  type bHYPRE_ParCSRDiagScale_6d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParCSRDiagScale_6d

  type bHYPRE_ParCSRDiagScale_7d
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type bHYPRE_ParCSRDiagScale_7d

end module bHYPRE_ParCSRDiagScale_type
