
#include "sidl_array_fAbbrev.h"
module sidl_array_array
  use sidl
  use sidl_array_type

  private :: &
    addRef_p

  interface addRef
    module procedure &
      addRef_p
  end interface

  private :: &
    deleteRef_p

  interface deleteRef
    module procedure &
      deleteRef_p
  end interface

  private :: &
    dimen_p

  interface dimen
    module procedure &
      dimen_p
  end interface

  private :: &
    type_p

  interface type
    module procedure &
      type_p
  end interface

  private :: &
    isColumnOrder_p

  interface isColumnOrder
    module procedure &
      isColumnOrder_p
  end interface

  private :: &
    isRowOrder_p

  interface isRowOrder
    module procedure &
      isRowOrder_p
  end interface

  private :: &
    is_null_p

  interface is_null
    module procedure &
      is_null_p
  end interface

  private :: &
    lower_p

  interface lower
    module procedure &
      lower_p
  end interface

  private :: &
    not_null_p

  interface not_null
    module procedure &
      not_null_p
  end interface

  private :: &
    set_null_p

  interface set_null
    module procedure &
      set_null_p
  end interface

  private :: &
    smartCopy_p

  interface smartCopy
    module procedure &
      smartCopy_p
  end interface

  private :: &
    stride_p

  interface stride
    module procedure &
      stride_p
  end interface

  private :: &
    upper_p

  interface upper
    module procedure &
      upper_p
  end interface

  private :: &
    length_p

  interface length
    module procedure &
      length_p
  end interface


contains


  subroutine smartCopy_p(src, dest)
    type(sidl__array), intent(in) :: src
    type(sidl__array), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl__array_smartCopy_m
    dim = 0
    call sidl__array_smartCopy_m(src, 0, dest)
  end subroutine smartCopy_p

  logical function  isColumnOrder_p(array)
    type(sidl__array), intent(in)  :: array
    external sidl__array_isColumnOrder_m
    call sidl__array_isColumnOrder_m(array, isColumnOrder_p)
  end function isColumnOrder_p

  logical function  isRowOrder_p(array)
    type(sidl__array), intent(in)  :: array
    external sidl__array_isRowOrder_m
    call sidl__array_isRowOrder_m(array, isRowOrder_p)
  end function isRowOrder_p

  integer (kind=sidl_int) function  dimen_p(array)
    type(sidl__array), intent(in)  :: array
    external sidl__array_dimen_m
    call sidl__array_dimen_m(array, dimen_p)
  end function dimen_p

  integer (kind=sidl_int) function  stride_p(array, index)
    type(sidl__array), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl__array_stride_m
    call sidl__array_stride_m(array, index, stride_p)
  end function stride_p

  integer (kind=sidl_int) function  lower_p(array, index)
    type(sidl__array), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl__array_lower_m
    call sidl__array_lower_m(array, index, lower_p)
  end function lower_p

  integer (kind=sidl_int) function  upper_p(array, index)
    type(sidl__array), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl__array_upper_m
    call sidl__array_upper_m(array, index, upper_p)
  end function upper_p

  integer (kind=sidl_int) function  length_p(array, index)
    type(sidl__array), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl__array_length_m
    call sidl__array_length_m(array, index, length_p)
  end function length_p

  integer (kind=sidl_int) function  type_p(array)
    type(sidl__array), intent(in)  :: array
    external sidl__array_type_m
    call sidl__array_type_m(array, type_p)
  end function type_p

  subroutine  addRef_p(array)
    type(sidl__array), intent(in)  :: array
    external sidl__array_addRef_m
    call sidl__array_addRef_m(array)
  end subroutine addRef_p

  subroutine  deleteRef_p(array)
    type(sidl__array), intent(in)  :: array
    external sidl__array_deleteRef_m
    call sidl__array_deleteRef_m(array)
  end subroutine deleteRef_p

  logical function is_null_p(array)
    type(sidl__array), intent(in) :: array
    is_null_p = (array%d_array .eq. 0)
  end function is_null_p

  logical function not_null_p(array)
    type(sidl__array), intent(in) :: array
    not_null_p = (array%d_array .ne. 0)
  end function not_null_p

  subroutine set_null_p(array)
    type(sidl__array), intent(out) :: array
    array%d_array = 0
  end subroutine set_null_p

end module sidl_array_array
