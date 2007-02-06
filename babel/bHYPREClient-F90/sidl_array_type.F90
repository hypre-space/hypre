#include "sidl_array_fAbbrev.h"

! 
! This file contains a FORTRAN 90 derived type for the
! sidl type sidl_array.
! 

module sidl_array_type
  use sidl
  type sidl__array
    sequence
    integer (kind=sidl_arrayptr) :: d_array
  end type sidl__array

end module sidl_array_type
