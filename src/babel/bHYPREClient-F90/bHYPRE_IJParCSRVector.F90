! 
! File:          bHYPRE_IJParCSRVector.F90
! Symbol:        bHYPRE.IJParCSRVector-v1.0.0
! Symbol Type:   class
! Babel Version: 1.0.0
! Description:   Client-side module for bHYPRE.IJParCSRVector
! 
! WARNING: Automatically generated; changes will be lost
! 
! 


! 
! Symbol "bHYPRE.IJParCSRVector" (version 1.0.0)
! 
! The IJParCSR vector class.
! 
! Objects of this type can be cast to IJVectorView or Vector
! objects using the {\tt \_\_cast} methods.
! 


#include "bHYPRE_IJParCSRVector_fAbbrev.h"
#include "bHYPRE_MPICommunicator_fAbbrev.h"
#include "sidl_ClassInfo_fAbbrev.h"
#include "bHYPRE_Vector_fAbbrev.h"
#include "bHYPRE_IJVectorView_fAbbrev.h"
#include "bHYPRE_ProblemDefinition_fAbbrev.h"
#include "sidl_BaseInterface_fAbbrev.h"
#include "sidl_RuntimeException_fAbbrev.h"
#include "bHYPRE_MatrixVectorView_fAbbrev.h"
#include "sidl_BaseException_fAbbrev.h"
#include "sidl_BaseClass_fAbbrev.h"

module bHYPRE_IJParCSRVector

  use sidl
  use bHYPRE_IJParCSRVector_type
  use bHYPRE_MPICommunicator_type
  use sidl_ClassInfo_type
  use bHYPRE_Vector_type
  use bHYPRE_IJVectorView_type
  use bHYPRE_ProblemDefinition_type
  use sidl_BaseInterface_type
  use sidl_RuntimeException_type
  use bHYPRE_MatrixVectorView_type
  use sidl_BaseException_type
  use sidl_BaseClass_type
  use sidl_rmi_Call_type
  use sidl_rmi_Return_type
  use sidl_rmi_Ticket_type
  use sidl_double_array
  use sidl_int_array

  private :: cast_0, cast_1, cast_2, cast_3, cast_4, cast_5, cast_6, cast_7,   &
    cast_8, cast_9, cast_10, cast_11
interface cast
  module procedure cast_0, cast_1, cast_2, cast_3, cast_4, cast_5, cast_6,     &
    cast_7, cast_8, cast_9, cast_10, cast_11
end interface

  private :: Create_s


interface Create
  module procedure Create_s
end interface
  private :: SetLocalRange_s


interface SetLocalRange
  module procedure SetLocalRange_s
end interface
  private :: SetValues_1s, &
  SetValues_2s


interface SetValues
  module procedure SetValues_1s, &
  SetValues_2s
end interface
  private :: AddToValues_1s, &
  AddToValues_2s


interface AddToValues
  module procedure AddToValues_1s, &
  AddToValues_2s
end interface
  private :: GetLocalRange_s


interface GetLocalRange
  module procedure GetLocalRange_s
end interface
  private :: GetValues_1s, &
  GetValues_2s


interface GetValues
  module procedure GetValues_1s, &
  GetValues_2s
end interface
  private :: Print_s


interface Print
  module procedure Print_s
end interface
  private :: Read_s


interface Read
  module procedure Read_s
end interface
  private :: SetCommunicator_s


interface SetCommunicator
  module procedure SetCommunicator_s
end interface
  private :: Destroy_s


interface Destroy
  module procedure Destroy_s
end interface
  private :: Initialize_s


interface Initialize
  module procedure Initialize_s
end interface
  private :: Assemble_s


interface Assemble
  module procedure Assemble_s
end interface
  private :: Clear_s


interface Clear
  module procedure Clear_s
end interface
  private :: Copy_s


interface Copy
  module procedure Copy_s
end interface
  private :: Clone_s


interface Clone
  module procedure Clone_s
end interface
  private :: Scale_s


interface Scale
  module procedure Scale_s
end interface
  private :: Dot_s


interface Dot
  module procedure Dot_s
end interface
  private :: Axpy_s


interface Axpy
  module procedure Axpy_s
end interface
  private :: newLocal_s, &
  newRemote_s


interface new
  module procedure newLocal_s, &
  newRemote_s
end interface
private :: rConnect_s
interface rConnect
  module procedure rConnect_s
end interface

  private :: addRef_s


interface addRef
  module procedure addRef_s
end interface
  private :: deleteRef_s


interface deleteRef
  module procedure deleteRef_s
end interface
  private :: isSame_s


interface isSame
  module procedure isSame_s
end interface
  private :: isType_s


interface isType
  module procedure isType_s
end interface
  private :: getClassInfo_s


interface getClassInfo
  module procedure getClassInfo_s
end interface

private :: exec_s
interface exec
  module procedure exec_s
end interface


private :: getURL_s
interface getURL
  module procedure getURL_s
end interface


private :: isRemote_s
interface isRemote
  module procedure isRemote_s
end interface


private :: isLocal_s
interface isLocal
  module procedure isLocal_s
end interface


private :: set_hooks_s
interface set_hooks
  module procedure set_hooks_s
end interface


private :: set_hooks_static_s
interface bHYPRE_IJParCSRVector__set_hooks_static
  module procedure set_hooks_static_s
end interface

private :: not_null_s
interface not_null
  module procedure not_null_s
end interface

private :: is_null_s
interface is_null
  module procedure is_null_s
end interface

private :: set_null_s
interface set_null
  module procedure set_null_s
end interface


contains



  recursive subroutine Create_s(mpi_comm, jlower, jupper, retval, exception)
    implicit none
    !  in bHYPRE.MPICommunicator mpi_comm
    type(bHYPRE_MPICommunicator_t) , intent(in) :: mpi_comm
    !  in int jlower
    integer (kind=sidl_int) , intent(in) :: jlower
    !  in int jupper
    integer (kind=sidl_int) , intent(in) :: jupper
    !  out bHYPRE.IJParCSRVector retval
    type(bHYPRE_IJParCSRVector_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Create_m
    call bHYPRE_IJParCSRVector_Create_m(mpi_comm, jlower, jupper, retval,      &
      exception)

  end subroutine Create_s


  recursive subroutine SetLocalRange_s(self, jlower, jupper, retval,           &
    exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in int jlower
    integer (kind=sidl_int) , intent(in) :: jlower
    !  in int jupper
    integer (kind=sidl_int) , intent(in) :: jupper
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_SetLocalRange_m
    call bHYPRE_IJParCSRVector_SetLocalRange_m(self, jlower, jupper, retval,   &
      exception)

  end subroutine SetLocalRange_s


  recursive subroutine SetValues_1s(self, indices, values, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in array<int,column-major> indices
    type(sidl_int_1d) , intent(in) :: indices
    !  in array<double,column-major> values
    type(sidl_double_1d) , intent(in) :: values
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_SetValues_a
    call bHYPRE_IJParCSRVector_SetValues_a(self, indices, values, retval,      &
      exception)

  end subroutine SetValues_1s

  recursive subroutine SetValues_2s(self, indices, values, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in rarray<int> indices(nvalues)
    integer (kind=sidl_int) , intent(in), dimension(:) :: indices
    !  in rarray<double> values(nvalues)
    real (kind=sidl_double) , intent(in), dimension(:) :: values
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  in int nvalues
    integer (kind=sidl_int) :: nvalues

    external bHYPRE_IJParCSRVector_SetValues_m
    nvalues = size(indices, 1)
    call bHYPRE_IJParCSRVector_SetValues_m(self, nvalues, indices, values,     &
      retval, exception)

  end subroutine SetValues_2s


  recursive subroutine AddToValues_1s(self, indices, values, retval,           &
    exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in array<int,column-major> indices
    type(sidl_int_1d) , intent(in) :: indices
    !  in array<double,column-major> values
    type(sidl_double_1d) , intent(in) :: values
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_AddToValues_a
    call bHYPRE_IJParCSRVector_AddToValues_a(self, indices, values, retval,    &
      exception)

  end subroutine AddToValues_1s

  recursive subroutine AddToValues_2s(self, indices, values, retval,           &
    exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in rarray<int> indices(nvalues)
    integer (kind=sidl_int) , intent(in), dimension(:) :: indices
    !  in rarray<double> values(nvalues)
    real (kind=sidl_double) , intent(in), dimension(:) :: values
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  in int nvalues
    integer (kind=sidl_int) :: nvalues

    external bHYPRE_IJParCSRVector_AddToValues_m
    nvalues = size(indices, 1)
    call bHYPRE_IJParCSRVector_AddToValues_m(self, nvalues, indices, values,   &
      retval, exception)

  end subroutine AddToValues_2s


  recursive subroutine GetLocalRange_s(self, jlower, jupper, retval,           &
    exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out int jlower
    integer (kind=sidl_int) , intent(out) :: jlower
    !  out int jupper
    integer (kind=sidl_int) , intent(out) :: jupper
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_GetLocalRange_m
    call bHYPRE_IJParCSRVector_GetLocalRange_m(self, jlower, jupper, retval,   &
      exception)

  end subroutine GetLocalRange_s


  recursive subroutine GetValues_1s(self, indices, values, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in array<int,column-major> indices
    type(sidl_int_1d) , intent(in) :: indices
    !  inout array<double,column-major> values
    type(sidl_double_1d) , intent(inout) :: values
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_GetValues_a
    call bHYPRE_IJParCSRVector_GetValues_a(self, indices, values, retval,      &
      exception)

  end subroutine GetValues_1s

  recursive subroutine GetValues_2s(self, indices, values, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in rarray<int> indices(nvalues)
    integer (kind=sidl_int) , intent(in), dimension(:) :: indices
    !  inout rarray<double> values(nvalues)
    real (kind=sidl_double) , intent(inout), dimension(:) :: values
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception
    !  in int nvalues
    integer (kind=sidl_int) :: nvalues

    external bHYPRE_IJParCSRVector_GetValues_m
    nvalues = size(indices, 1)
    call bHYPRE_IJParCSRVector_GetValues_m(self, nvalues, indices, values,     &
      retval, exception)

  end subroutine GetValues_2s


  recursive subroutine Print_s(self, filename, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in string filename
    character (len=*) , intent(in) :: filename
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Print_m
    call bHYPRE_IJParCSRVector_Print_m(self, filename, retval, exception)

  end subroutine Print_s


  recursive subroutine Read_s(self, filename, comm, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in string filename
    character (len=*) , intent(in) :: filename
    !  in bHYPRE.MPICommunicator comm
    type(bHYPRE_MPICommunicator_t) , intent(in) :: comm
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Read_m
    call bHYPRE_IJParCSRVector_Read_m(self, filename, comm, retval, exception)

  end subroutine Read_s


  recursive subroutine SetCommunicator_s(self, mpi_comm, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in bHYPRE.MPICommunicator mpi_comm
    type(bHYPRE_MPICommunicator_t) , intent(in) :: mpi_comm
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_SetCommunicator_m
    call bHYPRE_IJParCSRVector_SetCommunicator_m(self, mpi_comm, retval,       &
      exception)

  end subroutine SetCommunicator_s


  recursive subroutine Destroy_s(self, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Destroy_m
    call bHYPRE_IJParCSRVector_Destroy_m(self, exception)

  end subroutine Destroy_s


  recursive subroutine Initialize_s(self, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Initialize_m
    call bHYPRE_IJParCSRVector_Initialize_m(self, retval, exception)

  end subroutine Initialize_s


  recursive subroutine Assemble_s(self, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Assemble_m
    call bHYPRE_IJParCSRVector_Assemble_m(self, retval, exception)

  end subroutine Assemble_s


  recursive subroutine Clear_s(self, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Clear_m
    call bHYPRE_IJParCSRVector_Clear_m(self, retval, exception)

  end subroutine Clear_s


  recursive subroutine Copy_s(self, x, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in bHYPRE.Vector x
    type(bHYPRE_Vector_t) , intent(in) :: x
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Copy_m
    call bHYPRE_IJParCSRVector_Copy_m(self, x, retval, exception)

  end subroutine Copy_s


  recursive subroutine Clone_s(self, x, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out bHYPRE.Vector x
    type(bHYPRE_Vector_t) , intent(out) :: x
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Clone_m
    call bHYPRE_IJParCSRVector_Clone_m(self, x, retval, exception)

  end subroutine Clone_s


  recursive subroutine Scale_s(self, a, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in double a
    real (kind=sidl_double) , intent(in) :: a
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Scale_m
    call bHYPRE_IJParCSRVector_Scale_m(self, a, retval, exception)

  end subroutine Scale_s


  recursive subroutine Dot_s(self, x, d, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in bHYPRE.Vector x
    type(bHYPRE_Vector_t) , intent(in) :: x
    !  out double d
    real (kind=sidl_double) , intent(out) :: d
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Dot_m
    call bHYPRE_IJParCSRVector_Dot_m(self, x, d, retval, exception)

  end subroutine Dot_s


  recursive subroutine Axpy_s(self, a, x, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in double a
    real (kind=sidl_double) , intent(in) :: a
    !  in bHYPRE.Vector x
    type(bHYPRE_Vector_t) , intent(in) :: x
    !  out int retval
    integer (kind=sidl_int) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_Axpy_m
    call bHYPRE_IJParCSRVector_Axpy_m(self, a, x, retval, exception)

  end subroutine Axpy_s


  recursive subroutine newLocal_s(retval, exception)
    implicit none
    !  out bHYPRE.IJParCSRVector retval
    type(bHYPRE_IJParCSRVector_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_newLocal_m
    call bHYPRE_IJParCSRVector_newLocal_m(retval, exception)

  end subroutine newLocal_s


  recursive subroutine newRemote_s(self, url, exception)
    implicit none
    !  out bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(out) :: self
    !  in string url
    character (len=*) , intent(in) :: url
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_newRemote_m
    call bHYPRE_IJParCSRVector_newRemote_m(self, url, exception)

  end subroutine newRemote_s


  recursive subroutine rConnect_s(self, url, exception)
    implicit none
    !  out bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(out) :: self
    !  in string url
    character (len=*) , intent(in) :: url
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_rConnect_m
    call bHYPRE_IJParCSRVector_rConnect_m(self, url, exception)

  end subroutine rConnect_s


  recursive subroutine addRef_s(self, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_addRef_m
    call bHYPRE_IJParCSRVector_addRef_m(self, exception)

  end subroutine addRef_s


  recursive subroutine deleteRef_s(self, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_deleteRef_m
    call bHYPRE_IJParCSRVector_deleteRef_m(self, exception)

  end subroutine deleteRef_s


  recursive subroutine isSame_s(self, iobj, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in sidl.BaseInterface iobj
    type(sidl_BaseInterface_t) , intent(in) :: iobj
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_isSame_m
    call bHYPRE_IJParCSRVector_isSame_m(self, iobj, retval, exception)

  end subroutine isSame_s


  recursive subroutine isType_s(self, name, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in string name
    character (len=*) , intent(in) :: name
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_isType_m
    call bHYPRE_IJParCSRVector_isType_m(self, name, retval, exception)

  end subroutine isType_s


  recursive subroutine getClassInfo_s(self, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out sidl.ClassInfo retval
    type(sidl_ClassInfo_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector_getClassInfo_m
    call bHYPRE_IJParCSRVector_getClassInfo_m(self, retval, exception)

  end subroutine getClassInfo_s

  ! 
  ! Static function to cast from bHYPRE.IJParCSRVector
  ! to bHYPRE.IJVectorView.
  ! 

  subroutine cast_0(oldType, newType, exception)
    implicit none
    type(bHYPRE_IJParCSRVector_t), intent(in) :: oldType
    type(bHYPRE_IJVectorView_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_IJVectorView__cast_m

    call bHYPRE_IJVectorView__cast_m(oldType, newType, exception)
  end subroutine cast_0

  ! 
  ! Static function to cast from bHYPRE.IJVectorView
  ! to bHYPRE.IJParCSRVector.
  ! 

  subroutine cast_1(oldType, newType, exception)
    implicit none
    type(bHYPRE_IJVectorView_t), intent(in) :: oldType
    type(bHYPRE_IJParCSRVector_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_IJParCSRVector__cast_m

    call bHYPRE_IJParCSRVector__cast_m(oldType, newType, exception)
  end subroutine cast_1

  ! 
  ! Static function to cast from bHYPRE.IJParCSRVector
  ! to bHYPRE.MatrixVectorView.
  ! 

  subroutine cast_2(oldType, newType, exception)
    implicit none
    type(bHYPRE_IJParCSRVector_t), intent(in) :: oldType
    type(bHYPRE_MatrixVectorView_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_MatrixVectorView__cast_m

    call bHYPRE_MatrixVectorView__cast_m(oldType, newType, exception)
  end subroutine cast_2

  ! 
  ! Static function to cast from bHYPRE.MatrixVectorView
  ! to bHYPRE.IJParCSRVector.
  ! 

  subroutine cast_3(oldType, newType, exception)
    implicit none
    type(bHYPRE_MatrixVectorView_t), intent(in) :: oldType
    type(bHYPRE_IJParCSRVector_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_IJParCSRVector__cast_m

    call bHYPRE_IJParCSRVector__cast_m(oldType, newType, exception)
  end subroutine cast_3

  ! 
  ! Static function to cast from bHYPRE.IJParCSRVector
  ! to bHYPRE.ProblemDefinition.
  ! 

  subroutine cast_4(oldType, newType, exception)
    implicit none
    type(bHYPRE_IJParCSRVector_t), intent(in) :: oldType
    type(bHYPRE_ProblemDefinition_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_ProblemDefinition__cast_m

    call bHYPRE_ProblemDefinition__cast_m(oldType, newType, exception)
  end subroutine cast_4

  ! 
  ! Static function to cast from bHYPRE.ProblemDefinition
  ! to bHYPRE.IJParCSRVector.
  ! 

  subroutine cast_5(oldType, newType, exception)
    implicit none
    type(bHYPRE_ProblemDefinition_t), intent(in) :: oldType
    type(bHYPRE_IJParCSRVector_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_IJParCSRVector__cast_m

    call bHYPRE_IJParCSRVector__cast_m(oldType, newType, exception)
  end subroutine cast_5

  ! 
  ! Static function to cast from bHYPRE.IJParCSRVector
  ! to bHYPRE.Vector.
  ! 

  subroutine cast_6(oldType, newType, exception)
    implicit none
    type(bHYPRE_IJParCSRVector_t), intent(in) :: oldType
    type(bHYPRE_Vector_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_Vector__cast_m

    call bHYPRE_Vector__cast_m(oldType, newType, exception)
  end subroutine cast_6

  ! 
  ! Static function to cast from bHYPRE.Vector
  ! to bHYPRE.IJParCSRVector.
  ! 

  subroutine cast_7(oldType, newType, exception)
    implicit none
    type(bHYPRE_Vector_t), intent(in) :: oldType
    type(bHYPRE_IJParCSRVector_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_IJParCSRVector__cast_m

    call bHYPRE_IJParCSRVector__cast_m(oldType, newType, exception)
  end subroutine cast_7

  ! 
  ! Static function to cast from bHYPRE.IJParCSRVector
  ! to sidl.BaseClass.
  ! 

  subroutine cast_8(oldType, newType, exception)
    implicit none
    type(bHYPRE_IJParCSRVector_t), intent(in) :: oldType
    type(sidl_BaseClass_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_BaseClass__cast_m

    call sidl_BaseClass__cast_m(oldType, newType, exception)
  end subroutine cast_8

  ! 
  ! Static function to cast from sidl.BaseClass
  ! to bHYPRE.IJParCSRVector.
  ! 

  subroutine cast_9(oldType, newType, exception)
    implicit none
    type(sidl_BaseClass_t), intent(in) :: oldType
    type(bHYPRE_IJParCSRVector_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_IJParCSRVector__cast_m

    call bHYPRE_IJParCSRVector__cast_m(oldType, newType, exception)
  end subroutine cast_9

  ! 
  ! Static function to cast from bHYPRE.IJParCSRVector
  ! to sidl.BaseInterface.
  ! 

  subroutine cast_10(oldType, newType, exception)
    implicit none
    type(bHYPRE_IJParCSRVector_t), intent(in) :: oldType
    type(sidl_BaseInterface_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_BaseInterface__cast_m

    call sidl_BaseInterface__cast_m(oldType, newType, exception)
  end subroutine cast_10

  ! 
  ! Static function to cast from sidl.BaseInterface
  ! to bHYPRE.IJParCSRVector.
  ! 

  subroutine cast_11(oldType, newType, exception)
    implicit none
    type(sidl_BaseInterface_t), intent(in) :: oldType
    type(bHYPRE_IJParCSRVector_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external bHYPRE_IJParCSRVector__cast_m

    call bHYPRE_IJParCSRVector__cast_m(oldType, newType, exception)
  end subroutine cast_11


  recursive subroutine exec_s(self, methodName, inArgs, outArgs, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in string methodName
    character (len=*) , intent(in) :: methodName
    !  in sidl.rmi.Call inArgs
    type(sidl_rmi_Call_t) , intent(in) :: inArgs
    !  in sidl.rmi.Return outArgs
    type(sidl_rmi_Return_t) , intent(in) :: outArgs
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector__exec_m
    call bHYPRE_IJParCSRVector__exec_m(self, methodName, inArgs, outArgs,      &
      exception)

  end subroutine exec_s

  recursive subroutine getURL_s(self, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out string retval
    character (len=*) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector__getURL_m
    call bHYPRE_IJParCSRVector__getURL_m(self, retval, exception)

  end subroutine getURL_s

  recursive subroutine isRemote_s(self, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector__isRemote_m
    call bHYPRE_IJParCSRVector__isRemote_m(self, retval, exception)

  end subroutine isRemote_s

  recursive subroutine isLocal_s(self, retval, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector__isLocal_m
    call bHYPRE_IJParCSRVector__isLocal_m(self, retval, exception)

  end subroutine isLocal_s

  recursive subroutine set_hooks_s(self, on, exception)
    implicit none
    !  in bHYPRE.IJParCSRVector self
    type(bHYPRE_IJParCSRVector_t) , intent(in) :: self
    !  in bool on
    logical , intent(in) :: on
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector__set_hooks_m
    call bHYPRE_IJParCSRVector__set_hooks_m(self, on, exception)

  end subroutine set_hooks_s

  recursive subroutine set_hooks_static_s(on, exception)
    implicit none
    !  in bool on
    logical , intent(in) :: on
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external bHYPRE_IJParCSRVector__set_hooks_static_m
    call bHYPRE_IJParCSRVector__set_hooks_static_m(on, exception)

  end subroutine set_hooks_static_s
  logical function is_null_s(ext)
    type(bHYPRE_IJParCSRVector_t), intent(in) :: ext
    is_null_s = (ext%d_ior .eq. 0)
  end function is_null_s

  logical function not_null_s(ext)
    type(bHYPRE_IJParCSRVector_t), intent(in) :: ext
    not_null_s = (ext%d_ior .ne. 0)
  end function not_null_s

  subroutine set_null_s(ext)
    type(bHYPRE_IJParCSRVector_t), intent(out) :: ext
    ext%d_ior = 0
  end subroutine set_null_s


end module bHYPRE_IJParCSRVector

module bHYPRE_IJParCSRVector_array
  use sidl
  use bHYPRE_IJParCSRVector_type
  use sidl_array_type

  private :: &
    copy1_p, &
    copy2_p, &
    copy3_p, &
    copy4_p, &
    copy5_p, &
    copy6_p, &
    copy7_p

  interface copy
    module procedure &
      copy1_p, &
      copy2_p, &
      copy3_p, &
      copy4_p, &
      copy5_p, &
      copy6_p, &
      copy7_p
  end interface

  private :: &
    createCol1_p, &
    createCol2_p, &
    createCol3_p, &
    createCol4_p, &
    createCol5_p, &
    createCol6_p, &
    createCol7_p

  interface createCol
    module procedure &
      createCol1_p, &
      createCol2_p, &
      createCol3_p, &
      createCol4_p, &
      createCol5_p, &
      createCol6_p, &
      createCol7_p
  end interface

  private :: &
    createRow1_p, &
    createRow2_p, &
    createRow3_p, &
    createRow4_p, &
    createRow5_p, &
    createRow6_p, &
    createRow7_p

  interface createRow
    module procedure &
      createRow1_p, &
      createRow2_p, &
      createRow3_p, &
      createRow4_p, &
      createRow5_p, &
      createRow6_p, &
      createRow7_p
  end interface

  private :: &
    ensure1_p, &
    ensure2_p, &
    ensure3_p, &
    ensure4_p, &
    ensure5_p, &
    ensure6_p, &
    ensure7_p

  interface ensure
    module procedure &
      ensure1_p, &
      ensure2_p, &
      ensure3_p, &
      ensure4_p, &
      ensure5_p, &
      ensure6_p, &
      ensure7_p
  end interface

  private :: &
    addRef1_p, &
    addRef2_p, &
    addRef3_p, &
    addRef4_p, &
    addRef5_p, &
    addRef6_p, &
    addRef7_p

  interface addRef
    module procedure &
      addRef1_p, &
      addRef2_p, &
      addRef3_p, &
      addRef4_p, &
      addRef5_p, &
      addRef6_p, &
      addRef7_p
  end interface

  private :: &
    deleteRef1_p, &
    deleteRef2_p, &
    deleteRef3_p, &
    deleteRef4_p, &
    deleteRef5_p, &
    deleteRef6_p, &
    deleteRef7_p

  interface deleteRef
    module procedure &
      deleteRef1_p, &
      deleteRef2_p, &
      deleteRef3_p, &
      deleteRef4_p, &
      deleteRef5_p, &
      deleteRef6_p, &
      deleteRef7_p
  end interface

  private :: &
    dimen1_p, &
    dimen2_p, &
    dimen3_p, &
    dimen4_p, &
    dimen5_p, &
    dimen6_p, &
    dimen7_p

  interface dimen
    module procedure &
      dimen1_p, &
      dimen2_p, &
      dimen3_p, &
      dimen4_p, &
      dimen5_p, &
      dimen6_p, &
      dimen7_p
  end interface

  private :: &
    isColumnOrder1_p, &
    isColumnOrder2_p, &
    isColumnOrder3_p, &
    isColumnOrder4_p, &
    isColumnOrder5_p, &
    isColumnOrder6_p, &
    isColumnOrder7_p

  interface isColumnOrder
    module procedure &
      isColumnOrder1_p, &
      isColumnOrder2_p, &
      isColumnOrder3_p, &
      isColumnOrder4_p, &
      isColumnOrder5_p, &
      isColumnOrder6_p, &
      isColumnOrder7_p
  end interface

  private :: &
    isRowOrder1_p, &
    isRowOrder2_p, &
    isRowOrder3_p, &
    isRowOrder4_p, &
    isRowOrder5_p, &
    isRowOrder6_p, &
    isRowOrder7_p

  interface isRowOrder
    module procedure &
      isRowOrder1_p, &
      isRowOrder2_p, &
      isRowOrder3_p, &
      isRowOrder4_p, &
      isRowOrder5_p, &
      isRowOrder6_p, &
      isRowOrder7_p
  end interface

  private :: &
    is_null1_p, &
    is_null2_p, &
    is_null3_p, &
    is_null4_p, &
    is_null5_p, &
    is_null6_p, &
    is_null7_p

  interface is_null
    module procedure &
      is_null1_p, &
      is_null2_p, &
      is_null3_p, &
      is_null4_p, &
      is_null5_p, &
      is_null6_p, &
      is_null7_p
  end interface

  private :: &
    lower1_p, &
    lower2_p, &
    lower3_p, &
    lower4_p, &
    lower5_p, &
    lower6_p, &
    lower7_p

  interface lower
    module procedure &
      lower1_p, &
      lower2_p, &
      lower3_p, &
      lower4_p, &
      lower5_p, &
      lower6_p, &
      lower7_p
  end interface

  private :: &
    not_null1_p, &
    not_null2_p, &
    not_null3_p, &
    not_null4_p, &
    not_null5_p, &
    not_null6_p, &
    not_null7_p

  interface not_null
    module procedure &
      not_null1_p, &
      not_null2_p, &
      not_null3_p, &
      not_null4_p, &
      not_null5_p, &
      not_null6_p, &
      not_null7_p
  end interface

  private :: &
    set_null1_p, &
    set_null2_p, &
    set_null3_p, &
    set_null4_p, &
    set_null5_p, &
    set_null6_p, &
    set_null7_p

  interface set_null
    module procedure &
      set_null1_p, &
      set_null2_p, &
      set_null3_p, &
      set_null4_p, &
      set_null5_p, &
      set_null6_p, &
      set_null7_p
  end interface

  private :: &
    smartCopy1_p, &
    smartCopy2_p, &
    smartCopy3_p, &
    smartCopy4_p, &
    smartCopy5_p, &
    smartCopy6_p, &
    smartCopy7_p

  interface smartCopy
    module procedure &
      smartCopy1_p, &
      smartCopy2_p, &
      smartCopy3_p, &
      smartCopy4_p, &
      smartCopy5_p, &
      smartCopy6_p, &
      smartCopy7_p
  end interface

  private :: &
    stride1_p, &
    stride2_p, &
    stride3_p, &
    stride4_p, &
    stride5_p, &
    stride6_p, &
    stride7_p

  interface stride
    module procedure &
      stride1_p, &
      stride2_p, &
      stride3_p, &
      stride4_p, &
      stride5_p, &
      stride6_p, &
      stride7_p
  end interface

  private :: &
    upper1_p, &
    upper2_p, &
    upper3_p, &
    upper4_p, &
    upper5_p, &
    upper6_p, &
    upper7_p

  interface upper
    module procedure &
      upper1_p, &
      upper2_p, &
      upper3_p, &
      upper4_p, &
      upper5_p, &
      upper6_p, &
      upper7_p
  end interface

  private :: &
    length1_p, &
    length2_p, &
    length3_p, &
    length4_p, &
    length5_p, &
    length6_p, &
    length7_p

  interface length
    module procedure &
      length1_p, &
      length2_p, &
      length3_p, &
      length4_p, &
      length5_p, &
      length6_p, &
      length7_p
  end interface

  private :: create1d1_p

  interface create1d
    module procedure create1d1_p
  end interface

  private :: create2dRow2_p

  interface create2dRow
    module procedure create2dRow2_p
  end interface

  private :: create2dCol2_p

  interface create2dCol
    module procedure create2dCol2_p
  end interface

  private :: &
    slice11_p, &
    slice12_p, &
    slice22_p, &
    slice13_p, &
    slice23_p, &
    slice33_p, &
    slice14_p, &
    slice24_p, &
    slice34_p, &
    slice44_p, &
    slice15_p, &
    slice25_p, &
    slice35_p, &
    slice45_p, &
    slice55_p, &
    slice16_p, &
    slice26_p, &
    slice36_p, &
    slice46_p, &
    slice56_p, &
    slice66_p, &
    slice17_p, &
    slice27_p, &
    slice37_p, &
    slice47_p, &
    slice57_p, &
    slice67_p, &
    slice77_p

  interface slice
    module procedure &
      slice11_p, &
      slice12_p, &
      slice22_p, &
      slice13_p, &
      slice23_p, &
      slice33_p, &
      slice14_p, &
      slice24_p, &
      slice34_p, &
      slice44_p, &
      slice15_p, &
      slice25_p, &
      slice35_p, &
      slice45_p, &
      slice55_p, &
      slice16_p, &
      slice26_p, &
      slice36_p, &
      slice46_p, &
      slice56_p, &
      slice66_p, &
      slice17_p, &
      slice27_p, &
      slice37_p, &
      slice47_p, &
      slice57_p, &
      slice67_p, &
      slice77_p
  end interface

  private :: &
    getg1_p, &
    getg2_p, &
    getg3_p, &
    getg4_p, &
    getg5_p, &
    getg6_p, &
    getg7_p

  private :: &
    get1_p, &
    get2_p, &
    get3_p, &
    get4_p, &
    get5_p, &
    get6_p, &
    get7_p

  interface get
    module procedure &
      getg1_p, &
      get1_p, &
      getg2_p, &
      get2_p, &
      getg3_p, &
      get3_p, &
      getg4_p, &
      get4_p, &
      getg5_p, &
      get5_p, &
      getg6_p, &
      get6_p, &
      getg7_p, &
    get7_p
  end interface

  private :: &
    setg1_p, &
    setg2_p, &
    setg3_p, &
    setg4_p, &
    setg5_p, &
    setg6_p, &
    setg7_p

  private :: &
    set1_p, &
    set2_p, &
    set3_p, &
    set4_p, &
    set5_p, &
    set6_p, &
    set7_p

  interface set
    module procedure &
      setg1_p, &
      set1_p, &
      setg2_p, &
      set2_p, &
      setg3_p, &
      set3_p, &
      setg4_p, &
      set4_p, &
      setg5_p, &
      set5_p, &
      setg6_p, &
      set6_p, &
      setg7_p, &
    set7_p
  end interface


  private :: &
    castbHYPRE_IJParCSR298u0rugph_p, &
    castbHYPRE_IJParCSRzrwq18k_8n_p, &
    castbHYPRE_IJParCSRdtziqvrqbl_p, &
    castbHYPRE_IJParCSRo2mpwipbi1_p, &
    castbHYPRE_IJParCSRqsdye12e18_p, &
    castbHYPRE_IJParCSRz2vce5t9pb_p, &
    castbHYPRE_IJParCSRps7nodxcs3_p
interface cast
  module procedure &
    castbHYPRE_IJParCSR298u0rugph_p, &
    castbHYPRE_IJParCSRzrwq18k_8n_p, &
    castbHYPRE_IJParCSRdtziqvrqbl_p, &
    castbHYPRE_IJParCSRo2mpwipbi1_p, &
    castbHYPRE_IJParCSRqsdye12e18_p, &
    castbHYPRE_IJParCSRz2vce5t9pb_p, &
    castbHYPRE_IJParCSRps7nodxcs3_p
end interface


contains


  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: array
    external IJ_ary_createCol25wel5c_zwpeh_m
    call IJ_ary_createCol25wel5c_zwpeh_m(1, lower, upper, array)
  end subroutine createCol1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: array
    external IJ_ary_createRowzlt7a3v3chqp7_m
    call IJ_ary_createRowzlt7a3v3chqp7_m(1, lower, upper, array)
  end subroutine createRow1_p

  subroutine create1d1_p(len, array)
    integer (kind=sidl_int), intent(in) :: len
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: array
    external IJP_ary_create1dfkhji48mzt4r2_m
    call IJP_ary_create1dfkhji48mzt4r2_m(len, array)
  end subroutine create1d1_p

  subroutine copy1_p(src, dest)
    type(bHYPRE_IJParCSRVector_1d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_1d), intent(in) :: dest
    external IJParCS_ary_copyvmtx2vgf4n8uj_m
    call IJParCS_ary_copyvmtx2vgf4n8uj_m(src, dest)
  end subroutine copy1_p

  subroutine ensure1_p(src, dim, ordering, result)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: src
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external IJPar_ary_ensurewubyulbonik8u_m
    call IJPar_ary_ensurewubyulbonik8u_m(src, 1, ordering, result)
  end subroutine ensure1_p

  subroutine slice11_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: src
    integer (kind=sidl_int), dimension(1), intent(in) :: numElem
    integer (kind=sidl_int), dimension(1), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 1, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice11_p

  subroutine getg1_p(array, index, value)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCSR_ary_getyqyxev3rc58d4_m
    call IJParCSR_ary_getyqyxev3rc58d4_m(array, index, value)
  end subroutine getg1_p

  subroutine setg1_p(array, index, value)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCSR_ary_setwnbr3xycfmysw_m
    call IJParCSR_ary_setwnbr3xycfmysw_m(array, index, value)
  end subroutine setg1_p

  subroutine get1_p(array, &
      i1, &
      value)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCS_ary_get1n666nlfvwyxcs_m
    call IJParCS_ary_get1n666nlfvwyxcs_m(array, &
      i1, &
      value)
  end subroutine get1_p

  subroutine set1_p(array, &
      i1, &
      value)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCS_ary_set1ahl4vc9l5m2xc_m
    call IJParCS_ary_set1ahl4vc9l5m2xc_m(array, &
      i1, &
      value)
  end subroutine set1_p

  subroutine smartCopy1_p(src, dest)
    type(bHYPRE_IJParCSRVector_1d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: dest
    integer(sidl_int) :: dim
    external IJ_ary_smartCopyff9xvv143di44_m
    dim = 1
    call IJ_ary_smartCopyff9xvv143di44_m(src, 1, dest)
  end subroutine smartCopy1_p

  logical function  isColumnOrder1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    external ary_isColumnOrderu_4ej8107xlh_m
    call ary_isColumnOrderu_4ej8107xlh_m(array, isColumnOrder1_p)
  end function isColumnOrder1_p

  logical function  isRowOrder1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    external I_ary_isRowOrderf7_hpeg0frxm8_m
    call I_ary_isRowOrderf7_hpeg0frxm8_m(array, isRowOrder1_p)
  end function isRowOrder1_p

  integer (kind=sidl_int) function  dimen1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    external IJParC_ary_dimenro0yovyd3lmq3_m
    call IJParC_ary_dimenro0yovyd3lmq3_m(array, dimen1_p)
  end function dimen1_p

  integer (kind=sidl_int) function  stride1_p(array, index)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_stridebzdwhd_bv5_ka_m
    call IJPar_ary_stridebzdwhd_bv5_ka_m(array, index, stride1_p)
  end function stride1_p

  integer (kind=sidl_int) function  lower1_p(array, index)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_lowerxx7khmmpcfr_5_m
    call IJParC_ary_lowerxx7khmmpcfr_5_m(array, index, lower1_p)
  end function lower1_p

  integer (kind=sidl_int) function  upper1_p(array, index)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_upper3hs0j_ehm7cwn_m
    call IJParC_ary_upper3hs0j_ehm7cwn_m(array, index, upper1_p)
  end function upper1_p

  integer (kind=sidl_int) function  length1_p(array, index)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_lengthyuj2b8g5ubcu5_m
    call IJPar_ary_lengthyuj2b8g5ubcu5_m(array, index, length1_p)
  end function length1_p

  subroutine  addRef1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    external IJPar_ary_addRefd5fmq2ek2sovp_m
    call IJPar_ary_addRefd5fmq2ek2sovp_m(array)
  end subroutine addRef1_p

  subroutine  deleteRef1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(in)  :: array
    external IJ_ary_deleteRef_nushhd2qbdpw_m
    call IJ_ary_deleteRef_nushhd2qbdpw_m(array)
  end subroutine deleteRef1_p

  logical function is_null1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(in) :: array
    is_null1_p = (array%d_array .eq. 0)
  end function is_null1_p

  logical function not_null1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(in) :: array
    not_null1_p = (array%d_array .ne. 0)
  end function not_null1_p

  subroutine set_null1_p(array)
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: array
    external IJ_ary_createCol25wel5c_zwpeh_m
    call IJ_ary_createCol25wel5c_zwpeh_m(2, lower, upper, array)
  end subroutine createCol2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: array
    external IJ_ary_createRowzlt7a3v3chqp7_m
    call IJ_ary_createRowzlt7a3v3chqp7_m(2, lower, upper, array)
  end subroutine createRow2_p

  subroutine create2dCol2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: array
    external ary_create2dColm8_ov2yl6t7z5w_m
    call ary_create2dColm8_ov2yl6t7z5w_m(m, n, array)
  end subroutine create2dCol2_p

  subroutine create2dRow2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: array
    external ary_create2dRowa76ktz4suo24xr_m
    call ary_create2dRowa76ktz4suo24xr_m(m, n, array)
  end subroutine create2dRow2_p

  subroutine copy2_p(src, dest)
    type(bHYPRE_IJParCSRVector_2d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_2d), intent(in) :: dest
    external IJParCS_ary_copyvmtx2vgf4n8uj_m
    call IJParCS_ary_copyvmtx2vgf4n8uj_m(src, dest)
  end subroutine copy2_p

  subroutine ensure2_p(src, dim, ordering, result)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: src
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external IJPar_ary_ensurewubyulbonik8u_m
    call IJPar_ary_ensurewubyulbonik8u_m(src, 2, ordering, result)
  end subroutine ensure2_p

  subroutine slice12_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 1, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice12_p

  subroutine slice22_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 2, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice22_p

  subroutine getg2_p(array, index, value)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCSR_ary_getyqyxev3rc58d4_m
    call IJParCSR_ary_getyqyxev3rc58d4_m(array, index, value)
  end subroutine getg2_p

  subroutine setg2_p(array, index, value)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCSR_ary_setwnbr3xycfmysw_m
    call IJParCSR_ary_setwnbr3xycfmysw_m(array, index, value)
  end subroutine setg2_p

  subroutine get2_p(array, &
      i1, &
      i2, &
      value)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCS_ary_get26ez9iekfm3a0a_m
    call IJParCS_ary_get26ez9iekfm3a0a_m(array, &
      i1, &
      i2, &
      value)
  end subroutine get2_p

  subroutine set2_p(array, &
      i1, &
      i2, &
      value)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCS_ary_set2bs6gj0k1ndnua_m
    call IJParCS_ary_set2bs6gj0k1ndnua_m(array, &
      i1, &
      i2, &
      value)
  end subroutine set2_p

  subroutine smartCopy2_p(src, dest)
    type(bHYPRE_IJParCSRVector_2d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: dest
    integer(sidl_int) :: dim
    external IJ_ary_smartCopyff9xvv143di44_m
    dim = 2
    call IJ_ary_smartCopyff9xvv143di44_m(src, 2, dest)
  end subroutine smartCopy2_p

  logical function  isColumnOrder2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    external ary_isColumnOrderu_4ej8107xlh_m
    call ary_isColumnOrderu_4ej8107xlh_m(array, isColumnOrder2_p)
  end function isColumnOrder2_p

  logical function  isRowOrder2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    external I_ary_isRowOrderf7_hpeg0frxm8_m
    call I_ary_isRowOrderf7_hpeg0frxm8_m(array, isRowOrder2_p)
  end function isRowOrder2_p

  integer (kind=sidl_int) function  dimen2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    external IJParC_ary_dimenro0yovyd3lmq3_m
    call IJParC_ary_dimenro0yovyd3lmq3_m(array, dimen2_p)
  end function dimen2_p

  integer (kind=sidl_int) function  stride2_p(array, index)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_stridebzdwhd_bv5_ka_m
    call IJPar_ary_stridebzdwhd_bv5_ka_m(array, index, stride2_p)
  end function stride2_p

  integer (kind=sidl_int) function  lower2_p(array, index)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_lowerxx7khmmpcfr_5_m
    call IJParC_ary_lowerxx7khmmpcfr_5_m(array, index, lower2_p)
  end function lower2_p

  integer (kind=sidl_int) function  upper2_p(array, index)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_upper3hs0j_ehm7cwn_m
    call IJParC_ary_upper3hs0j_ehm7cwn_m(array, index, upper2_p)
  end function upper2_p

  integer (kind=sidl_int) function  length2_p(array, index)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_lengthyuj2b8g5ubcu5_m
    call IJPar_ary_lengthyuj2b8g5ubcu5_m(array, index, length2_p)
  end function length2_p

  subroutine  addRef2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    external IJPar_ary_addRefd5fmq2ek2sovp_m
    call IJPar_ary_addRefd5fmq2ek2sovp_m(array)
  end subroutine addRef2_p

  subroutine  deleteRef2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(in)  :: array
    external IJ_ary_deleteRef_nushhd2qbdpw_m
    call IJ_ary_deleteRef_nushhd2qbdpw_m(array)
  end subroutine deleteRef2_p

  logical function is_null2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(in) :: array
    is_null2_p = (array%d_array .eq. 0)
  end function is_null2_p

  logical function not_null2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(in) :: array
    not_null2_p = (array%d_array .ne. 0)
  end function not_null2_p

  subroutine set_null2_p(array)
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: array
    external IJ_ary_createCol25wel5c_zwpeh_m
    call IJ_ary_createCol25wel5c_zwpeh_m(3, lower, upper, array)
  end subroutine createCol3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: array
    external IJ_ary_createRowzlt7a3v3chqp7_m
    call IJ_ary_createRowzlt7a3v3chqp7_m(3, lower, upper, array)
  end subroutine createRow3_p

  subroutine copy3_p(src, dest)
    type(bHYPRE_IJParCSRVector_3d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_3d), intent(in) :: dest
    external IJParCS_ary_copyvmtx2vgf4n8uj_m
    call IJParCS_ary_copyvmtx2vgf4n8uj_m(src, dest)
  end subroutine copy3_p

  subroutine ensure3_p(src, dim, ordering, result)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: src
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external IJPar_ary_ensurewubyulbonik8u_m
    call IJPar_ary_ensurewubyulbonik8u_m(src, 3, ordering, result)
  end subroutine ensure3_p

  subroutine slice13_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 1, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice13_p

  subroutine slice23_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 2, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice23_p

  subroutine slice33_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 3, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice33_p

  subroutine getg3_p(array, index, value)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCSR_ary_getyqyxev3rc58d4_m
    call IJParCSR_ary_getyqyxev3rc58d4_m(array, index, value)
  end subroutine getg3_p

  subroutine setg3_p(array, index, value)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCSR_ary_setwnbr3xycfmysw_m
    call IJParCSR_ary_setwnbr3xycfmysw_m(array, index, value)
  end subroutine setg3_p

  subroutine get3_p(array, &
      i1, &
      i2, &
      i3, &
      value)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCS_ary_get3hbhhskzbm7e7y_m
    call IJParCS_ary_get3hbhhskzbm7e7y_m(array, &
      i1, &
      i2, &
      i3, &
      value)
  end subroutine get3_p

  subroutine set3_p(array, &
      i1, &
      i2, &
      i3, &
      value)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCS_ary_set3wr8c2js49y45x_m
    call IJParCS_ary_set3wr8c2js49y45x_m(array, &
      i1, &
      i2, &
      i3, &
      value)
  end subroutine set3_p

  subroutine smartCopy3_p(src, dest)
    type(bHYPRE_IJParCSRVector_3d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: dest
    integer(sidl_int) :: dim
    external IJ_ary_smartCopyff9xvv143di44_m
    dim = 3
    call IJ_ary_smartCopyff9xvv143di44_m(src, 3, dest)
  end subroutine smartCopy3_p

  logical function  isColumnOrder3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    external ary_isColumnOrderu_4ej8107xlh_m
    call ary_isColumnOrderu_4ej8107xlh_m(array, isColumnOrder3_p)
  end function isColumnOrder3_p

  logical function  isRowOrder3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    external I_ary_isRowOrderf7_hpeg0frxm8_m
    call I_ary_isRowOrderf7_hpeg0frxm8_m(array, isRowOrder3_p)
  end function isRowOrder3_p

  integer (kind=sidl_int) function  dimen3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    external IJParC_ary_dimenro0yovyd3lmq3_m
    call IJParC_ary_dimenro0yovyd3lmq3_m(array, dimen3_p)
  end function dimen3_p

  integer (kind=sidl_int) function  stride3_p(array, index)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_stridebzdwhd_bv5_ka_m
    call IJPar_ary_stridebzdwhd_bv5_ka_m(array, index, stride3_p)
  end function stride3_p

  integer (kind=sidl_int) function  lower3_p(array, index)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_lowerxx7khmmpcfr_5_m
    call IJParC_ary_lowerxx7khmmpcfr_5_m(array, index, lower3_p)
  end function lower3_p

  integer (kind=sidl_int) function  upper3_p(array, index)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_upper3hs0j_ehm7cwn_m
    call IJParC_ary_upper3hs0j_ehm7cwn_m(array, index, upper3_p)
  end function upper3_p

  integer (kind=sidl_int) function  length3_p(array, index)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_lengthyuj2b8g5ubcu5_m
    call IJPar_ary_lengthyuj2b8g5ubcu5_m(array, index, length3_p)
  end function length3_p

  subroutine  addRef3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    external IJPar_ary_addRefd5fmq2ek2sovp_m
    call IJPar_ary_addRefd5fmq2ek2sovp_m(array)
  end subroutine addRef3_p

  subroutine  deleteRef3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(in)  :: array
    external IJ_ary_deleteRef_nushhd2qbdpw_m
    call IJ_ary_deleteRef_nushhd2qbdpw_m(array)
  end subroutine deleteRef3_p

  logical function is_null3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(in) :: array
    is_null3_p = (array%d_array .eq. 0)
  end function is_null3_p

  logical function not_null3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(in) :: array
    not_null3_p = (array%d_array .ne. 0)
  end function not_null3_p

  subroutine set_null3_p(array)
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: array
    external IJ_ary_createCol25wel5c_zwpeh_m
    call IJ_ary_createCol25wel5c_zwpeh_m(4, lower, upper, array)
  end subroutine createCol4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: array
    external IJ_ary_createRowzlt7a3v3chqp7_m
    call IJ_ary_createRowzlt7a3v3chqp7_m(4, lower, upper, array)
  end subroutine createRow4_p

  subroutine copy4_p(src, dest)
    type(bHYPRE_IJParCSRVector_4d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_4d), intent(in) :: dest
    external IJParCS_ary_copyvmtx2vgf4n8uj_m
    call IJParCS_ary_copyvmtx2vgf4n8uj_m(src, dest)
  end subroutine copy4_p

  subroutine ensure4_p(src, dim, ordering, result)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: src
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external IJPar_ary_ensurewubyulbonik8u_m
    call IJPar_ary_ensurewubyulbonik8u_m(src, 4, ordering, result)
  end subroutine ensure4_p

  subroutine slice14_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 1, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice14_p

  subroutine slice24_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 2, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice24_p

  subroutine slice34_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 3, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice34_p

  subroutine slice44_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 4, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice44_p

  subroutine getg4_p(array, index, value)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCSR_ary_getyqyxev3rc58d4_m
    call IJParCSR_ary_getyqyxev3rc58d4_m(array, index, value)
  end subroutine getg4_p

  subroutine setg4_p(array, index, value)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCSR_ary_setwnbr3xycfmysw_m
    call IJParCSR_ary_setwnbr3xycfmysw_m(array, index, value)
  end subroutine setg4_p

  subroutine get4_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCS_ary_get4_gy6fde7o1ff__m
    call IJParCS_ary_get4_gy6fde7o1ff__m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
  end subroutine get4_p

  subroutine set4_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCS_ary_set4ll5drlhe1pbpr_m
    call IJParCS_ary_set4ll5drlhe1pbpr_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
  end subroutine set4_p

  subroutine smartCopy4_p(src, dest)
    type(bHYPRE_IJParCSRVector_4d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: dest
    integer(sidl_int) :: dim
    external IJ_ary_smartCopyff9xvv143di44_m
    dim = 4
    call IJ_ary_smartCopyff9xvv143di44_m(src, 4, dest)
  end subroutine smartCopy4_p

  logical function  isColumnOrder4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    external ary_isColumnOrderu_4ej8107xlh_m
    call ary_isColumnOrderu_4ej8107xlh_m(array, isColumnOrder4_p)
  end function isColumnOrder4_p

  logical function  isRowOrder4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    external I_ary_isRowOrderf7_hpeg0frxm8_m
    call I_ary_isRowOrderf7_hpeg0frxm8_m(array, isRowOrder4_p)
  end function isRowOrder4_p

  integer (kind=sidl_int) function  dimen4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    external IJParC_ary_dimenro0yovyd3lmq3_m
    call IJParC_ary_dimenro0yovyd3lmq3_m(array, dimen4_p)
  end function dimen4_p

  integer (kind=sidl_int) function  stride4_p(array, index)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_stridebzdwhd_bv5_ka_m
    call IJPar_ary_stridebzdwhd_bv5_ka_m(array, index, stride4_p)
  end function stride4_p

  integer (kind=sidl_int) function  lower4_p(array, index)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_lowerxx7khmmpcfr_5_m
    call IJParC_ary_lowerxx7khmmpcfr_5_m(array, index, lower4_p)
  end function lower4_p

  integer (kind=sidl_int) function  upper4_p(array, index)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_upper3hs0j_ehm7cwn_m
    call IJParC_ary_upper3hs0j_ehm7cwn_m(array, index, upper4_p)
  end function upper4_p

  integer (kind=sidl_int) function  length4_p(array, index)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_lengthyuj2b8g5ubcu5_m
    call IJPar_ary_lengthyuj2b8g5ubcu5_m(array, index, length4_p)
  end function length4_p

  subroutine  addRef4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    external IJPar_ary_addRefd5fmq2ek2sovp_m
    call IJPar_ary_addRefd5fmq2ek2sovp_m(array)
  end subroutine addRef4_p

  subroutine  deleteRef4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(in)  :: array
    external IJ_ary_deleteRef_nushhd2qbdpw_m
    call IJ_ary_deleteRef_nushhd2qbdpw_m(array)
  end subroutine deleteRef4_p

  logical function is_null4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(in) :: array
    is_null4_p = (array%d_array .eq. 0)
  end function is_null4_p

  logical function not_null4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(in) :: array
    not_null4_p = (array%d_array .ne. 0)
  end function not_null4_p

  subroutine set_null4_p(array)
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: array
    external IJ_ary_createCol25wel5c_zwpeh_m
    call IJ_ary_createCol25wel5c_zwpeh_m(5, lower, upper, array)
  end subroutine createCol5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: array
    external IJ_ary_createRowzlt7a3v3chqp7_m
    call IJ_ary_createRowzlt7a3v3chqp7_m(5, lower, upper, array)
  end subroutine createRow5_p

  subroutine copy5_p(src, dest)
    type(bHYPRE_IJParCSRVector_5d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_5d), intent(in) :: dest
    external IJParCS_ary_copyvmtx2vgf4n8uj_m
    call IJParCS_ary_copyvmtx2vgf4n8uj_m(src, dest)
  end subroutine copy5_p

  subroutine ensure5_p(src, dim, ordering, result)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: src
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external IJPar_ary_ensurewubyulbonik8u_m
    call IJPar_ary_ensurewubyulbonik8u_m(src, 5, ordering, result)
  end subroutine ensure5_p

  subroutine slice15_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 1, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice15_p

  subroutine slice25_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 2, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice25_p

  subroutine slice35_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 3, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice35_p

  subroutine slice45_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 4, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice45_p

  subroutine slice55_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 5, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice55_p

  subroutine getg5_p(array, index, value)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCSR_ary_getyqyxev3rc58d4_m
    call IJParCSR_ary_getyqyxev3rc58d4_m(array, index, value)
  end subroutine getg5_p

  subroutine setg5_p(array, index, value)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCSR_ary_setwnbr3xycfmysw_m
    call IJParCSR_ary_setwnbr3xycfmysw_m(array, index, value)
  end subroutine setg5_p

  subroutine get5_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCS_ary_get5uqeftdeceu7gm_m
    call IJParCS_ary_get5uqeftdeceu7gm_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
  end subroutine get5_p

  subroutine set5_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCS_ary_set5z0ixcig8lmmcg_m
    call IJParCS_ary_set5z0ixcig8lmmcg_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
  end subroutine set5_p

  subroutine smartCopy5_p(src, dest)
    type(bHYPRE_IJParCSRVector_5d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: dest
    integer(sidl_int) :: dim
    external IJ_ary_smartCopyff9xvv143di44_m
    dim = 5
    call IJ_ary_smartCopyff9xvv143di44_m(src, 5, dest)
  end subroutine smartCopy5_p

  logical function  isColumnOrder5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    external ary_isColumnOrderu_4ej8107xlh_m
    call ary_isColumnOrderu_4ej8107xlh_m(array, isColumnOrder5_p)
  end function isColumnOrder5_p

  logical function  isRowOrder5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    external I_ary_isRowOrderf7_hpeg0frxm8_m
    call I_ary_isRowOrderf7_hpeg0frxm8_m(array, isRowOrder5_p)
  end function isRowOrder5_p

  integer (kind=sidl_int) function  dimen5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    external IJParC_ary_dimenro0yovyd3lmq3_m
    call IJParC_ary_dimenro0yovyd3lmq3_m(array, dimen5_p)
  end function dimen5_p

  integer (kind=sidl_int) function  stride5_p(array, index)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_stridebzdwhd_bv5_ka_m
    call IJPar_ary_stridebzdwhd_bv5_ka_m(array, index, stride5_p)
  end function stride5_p

  integer (kind=sidl_int) function  lower5_p(array, index)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_lowerxx7khmmpcfr_5_m
    call IJParC_ary_lowerxx7khmmpcfr_5_m(array, index, lower5_p)
  end function lower5_p

  integer (kind=sidl_int) function  upper5_p(array, index)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_upper3hs0j_ehm7cwn_m
    call IJParC_ary_upper3hs0j_ehm7cwn_m(array, index, upper5_p)
  end function upper5_p

  integer (kind=sidl_int) function  length5_p(array, index)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_lengthyuj2b8g5ubcu5_m
    call IJPar_ary_lengthyuj2b8g5ubcu5_m(array, index, length5_p)
  end function length5_p

  subroutine  addRef5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    external IJPar_ary_addRefd5fmq2ek2sovp_m
    call IJPar_ary_addRefd5fmq2ek2sovp_m(array)
  end subroutine addRef5_p

  subroutine  deleteRef5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(in)  :: array
    external IJ_ary_deleteRef_nushhd2qbdpw_m
    call IJ_ary_deleteRef_nushhd2qbdpw_m(array)
  end subroutine deleteRef5_p

  logical function is_null5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(in) :: array
    is_null5_p = (array%d_array .eq. 0)
  end function is_null5_p

  logical function not_null5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(in) :: array
    not_null5_p = (array%d_array .ne. 0)
  end function not_null5_p

  subroutine set_null5_p(array)
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_6d), intent(out) :: array
    external IJ_ary_createCol25wel5c_zwpeh_m
    call IJ_ary_createCol25wel5c_zwpeh_m(6, lower, upper, array)
  end subroutine createCol6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_6d), intent(out) :: array
    external IJ_ary_createRowzlt7a3v3chqp7_m
    call IJ_ary_createRowzlt7a3v3chqp7_m(6, lower, upper, array)
  end subroutine createRow6_p

  subroutine copy6_p(src, dest)
    type(bHYPRE_IJParCSRVector_6d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_6d), intent(in) :: dest
    external IJParCS_ary_copyvmtx2vgf4n8uj_m
    call IJParCS_ary_copyvmtx2vgf4n8uj_m(src, dest)
  end subroutine copy6_p

  subroutine ensure6_p(src, dim, ordering, result)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: src
    type(bHYPRE_IJParCSRVector_6d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external IJPar_ary_ensurewubyulbonik8u_m
    call IJPar_ary_ensurewubyulbonik8u_m(src, 6, ordering, result)
  end subroutine ensure6_p

  subroutine slice16_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 1, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice16_p

  subroutine slice26_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 2, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice26_p

  subroutine slice36_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 3, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice36_p

  subroutine slice46_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 4, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice46_p

  subroutine slice56_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 5, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice56_p

  subroutine slice66_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 6, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice66_p

  subroutine getg6_p(array, index, value)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCSR_ary_getyqyxev3rc58d4_m
    call IJParCSR_ary_getyqyxev3rc58d4_m(array, index, value)
  end subroutine getg6_p

  subroutine setg6_p(array, index, value)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCSR_ary_setwnbr3xycfmysw_m
    call IJParCSR_ary_setwnbr3xycfmysw_m(array, index, value)
  end subroutine setg6_p

  subroutine get6_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCS_ary_get6babx68b1yrofz_m
    call IJParCS_ary_get6babx68b1yrofz_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
  end subroutine get6_p

  subroutine set6_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCS_ary_set62ixkwmd7hut7w_m
    call IJParCS_ary_set62ixkwmd7hut7w_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
  end subroutine set6_p

  subroutine smartCopy6_p(src, dest)
    type(bHYPRE_IJParCSRVector_6d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_6d), intent(out) :: dest
    integer(sidl_int) :: dim
    external IJ_ary_smartCopyff9xvv143di44_m
    dim = 6
    call IJ_ary_smartCopyff9xvv143di44_m(src, 6, dest)
  end subroutine smartCopy6_p

  logical function  isColumnOrder6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    external ary_isColumnOrderu_4ej8107xlh_m
    call ary_isColumnOrderu_4ej8107xlh_m(array, isColumnOrder6_p)
  end function isColumnOrder6_p

  logical function  isRowOrder6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    external I_ary_isRowOrderf7_hpeg0frxm8_m
    call I_ary_isRowOrderf7_hpeg0frxm8_m(array, isRowOrder6_p)
  end function isRowOrder6_p

  integer (kind=sidl_int) function  dimen6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    external IJParC_ary_dimenro0yovyd3lmq3_m
    call IJParC_ary_dimenro0yovyd3lmq3_m(array, dimen6_p)
  end function dimen6_p

  integer (kind=sidl_int) function  stride6_p(array, index)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_stridebzdwhd_bv5_ka_m
    call IJPar_ary_stridebzdwhd_bv5_ka_m(array, index, stride6_p)
  end function stride6_p

  integer (kind=sidl_int) function  lower6_p(array, index)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_lowerxx7khmmpcfr_5_m
    call IJParC_ary_lowerxx7khmmpcfr_5_m(array, index, lower6_p)
  end function lower6_p

  integer (kind=sidl_int) function  upper6_p(array, index)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_upper3hs0j_ehm7cwn_m
    call IJParC_ary_upper3hs0j_ehm7cwn_m(array, index, upper6_p)
  end function upper6_p

  integer (kind=sidl_int) function  length6_p(array, index)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_lengthyuj2b8g5ubcu5_m
    call IJPar_ary_lengthyuj2b8g5ubcu5_m(array, index, length6_p)
  end function length6_p

  subroutine  addRef6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    external IJPar_ary_addRefd5fmq2ek2sovp_m
    call IJPar_ary_addRefd5fmq2ek2sovp_m(array)
  end subroutine addRef6_p

  subroutine  deleteRef6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(in)  :: array
    external IJ_ary_deleteRef_nushhd2qbdpw_m
    call IJ_ary_deleteRef_nushhd2qbdpw_m(array)
  end subroutine deleteRef6_p

  logical function is_null6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(in) :: array
    is_null6_p = (array%d_array .eq. 0)
  end function is_null6_p

  logical function not_null6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(in) :: array
    not_null6_p = (array%d_array .ne. 0)
  end function not_null6_p

  subroutine set_null6_p(array)
    type(bHYPRE_IJParCSRVector_6d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_7d), intent(out) :: array
    external IJ_ary_createCol25wel5c_zwpeh_m
    call IJ_ary_createCol25wel5c_zwpeh_m(7, lower, upper, array)
  end subroutine createCol7_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(bHYPRE_IJParCSRVector_7d), intent(out) :: array
    external IJ_ary_createRowzlt7a3v3chqp7_m
    call IJ_ary_createRowzlt7a3v3chqp7_m(7, lower, upper, array)
  end subroutine createRow7_p

  subroutine copy7_p(src, dest)
    type(bHYPRE_IJParCSRVector_7d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_7d), intent(in) :: dest
    external IJParCS_ary_copyvmtx2vgf4n8uj_m
    call IJParCS_ary_copyvmtx2vgf4n8uj_m(src, dest)
  end subroutine copy7_p

  subroutine ensure7_p(src, dim, ordering, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    type(bHYPRE_IJParCSRVector_7d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external IJPar_ary_ensurewubyulbonik8u_m
    call IJPar_ary_ensurewubyulbonik8u_m(src, 7, ordering, result)
  end subroutine ensure7_p

  subroutine slice17_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 1, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice17_p

  subroutine slice27_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 2, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice27_p

  subroutine slice37_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 3, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice37_p

  subroutine slice47_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 4, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice47_p

  subroutine slice57_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 5, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice57_p

  subroutine slice67_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 6, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice67_p

  subroutine slice77_p(src, numElem, srcStart, srcStride, newLower, result)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(bHYPRE_IJParCSRVector_7d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external IJParC_ary_slicefb5fbf5vdg0q__m
    call IJParC_ary_slicefb5fbf5vdg0q__m(src, 7, numElem, srcStart, srcStride, &
      newLower, result)
  end subroutine slice77_p

  subroutine getg7_p(array, index, value)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCSR_ary_getyqyxev3rc58d4_m
    call IJParCSR_ary_getyqyxev3rc58d4_m(array, index, value)
  end subroutine getg7_p

  subroutine setg7_p(array, index, value)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCSR_ary_setwnbr3xycfmysw_m
    call IJParCSR_ary_setwnbr3xycfmysw_m(array, index, value)
  end subroutine setg7_p

  subroutine get7_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      i7, &
      value)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(bHYPRE_IJParCSRVector_t), intent(out) :: value
    external IJParCS_ary_get7ufhdpc62vak_0_m
    call IJParCS_ary_get7ufhdpc62vak_0_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      i7, &
      value)
  end subroutine get7_p

  subroutine set7_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      i7, &
      value)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(bHYPRE_IJParCSRVector_t), intent(in) :: value
    external IJParCS_ary_set7xs8o5a_hr923i_m
    call IJParCS_ary_set7xs8o5a_hr923i_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      i7, &
      value)
  end subroutine set7_p

  subroutine smartCopy7_p(src, dest)
    type(bHYPRE_IJParCSRVector_7d), intent(in) :: src
    type(bHYPRE_IJParCSRVector_7d), intent(out) :: dest
    integer(sidl_int) :: dim
    external IJ_ary_smartCopyff9xvv143di44_m
    dim = 7
    call IJ_ary_smartCopyff9xvv143di44_m(src, 7, dest)
  end subroutine smartCopy7_p

  logical function  isColumnOrder7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    external ary_isColumnOrderu_4ej8107xlh_m
    call ary_isColumnOrderu_4ej8107xlh_m(array, isColumnOrder7_p)
  end function isColumnOrder7_p

  logical function  isRowOrder7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    external I_ary_isRowOrderf7_hpeg0frxm8_m
    call I_ary_isRowOrderf7_hpeg0frxm8_m(array, isRowOrder7_p)
  end function isRowOrder7_p

  integer (kind=sidl_int) function  dimen7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    external IJParC_ary_dimenro0yovyd3lmq3_m
    call IJParC_ary_dimenro0yovyd3lmq3_m(array, dimen7_p)
  end function dimen7_p

  integer (kind=sidl_int) function  stride7_p(array, index)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_stridebzdwhd_bv5_ka_m
    call IJPar_ary_stridebzdwhd_bv5_ka_m(array, index, stride7_p)
  end function stride7_p

  integer (kind=sidl_int) function  lower7_p(array, index)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_lowerxx7khmmpcfr_5_m
    call IJParC_ary_lowerxx7khmmpcfr_5_m(array, index, lower7_p)
  end function lower7_p

  integer (kind=sidl_int) function  upper7_p(array, index)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJParC_ary_upper3hs0j_ehm7cwn_m
    call IJParC_ary_upper3hs0j_ehm7cwn_m(array, index, upper7_p)
  end function upper7_p

  integer (kind=sidl_int) function  length7_p(array, index)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external IJPar_ary_lengthyuj2b8g5ubcu5_m
    call IJPar_ary_lengthyuj2b8g5ubcu5_m(array, index, length7_p)
  end function length7_p

  subroutine  addRef7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    external IJPar_ary_addRefd5fmq2ek2sovp_m
    call IJPar_ary_addRefd5fmq2ek2sovp_m(array)
  end subroutine addRef7_p

  subroutine  deleteRef7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(in)  :: array
    external IJ_ary_deleteRef_nushhd2qbdpw_m
    call IJ_ary_deleteRef_nushhd2qbdpw_m(array)
  end subroutine deleteRef7_p

  logical function is_null7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(in) :: array
    is_null7_p = (array%d_array .eq. 0)
  end function is_null7_p

  logical function not_null7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(in) :: array
    not_null7_p = (array%d_array .ne. 0)
  end function not_null7_p

  subroutine set_null7_p(array)
    type(bHYPRE_IJParCSRVector_7d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null7_p

  subroutine castbHYPRE_IJParCSR298u0rugph_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(bHYPRE_IJParCSRVector_1d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castbHYPRE_IJParCSR298u0rugph_p

  subroutine castbHYPRE_IJParCSRzrwq18k_8n_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(bHYPRE_IJParCSRVector_2d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castbHYPRE_IJParCSRzrwq18k_8n_p

  subroutine castbHYPRE_IJParCSRdtziqvrqbl_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(bHYPRE_IJParCSRVector_3d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castbHYPRE_IJParCSRdtziqvrqbl_p

  subroutine castbHYPRE_IJParCSRo2mpwipbi1_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(bHYPRE_IJParCSRVector_4d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castbHYPRE_IJParCSRo2mpwipbi1_p

  subroutine castbHYPRE_IJParCSRqsdye12e18_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(bHYPRE_IJParCSRVector_5d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castbHYPRE_IJParCSRqsdye12e18_p

  subroutine castbHYPRE_IJParCSRz2vce5t9pb_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(bHYPRE_IJParCSRVector_6d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castbHYPRE_IJParCSRz2vce5t9pb_p

  subroutine castbHYPRE_IJParCSRps7nodxcs3_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(bHYPRE_IJParCSRVector_7d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castbHYPRE_IJParCSRps7nodxcs3_p


end module bHYPRE_IJParCSRVector_array
