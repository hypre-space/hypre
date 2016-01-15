! 
! File:          sidl_rmi_Call.F90
! Symbol:        sidl.rmi.Call-v0.9.15
! Symbol Type:   interface
! Babel Version: 1.0.0
! Release:       $Name$
! Revision:      @(#) $Id$
! Description:   Client-side module for sidl.rmi.Call
! 
! Copyright (c) 2000-2002, The Regents of the University of California.
! Produced at the Lawrence Livermore National Laboratory.
! Written by the Components Team <components@llnl.gov>
! All rights reserved.
! 
! This file is part of Babel. For more information, see
! http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
! for Our Notice and the LICENSE file for the GNU Lesser General Public
! License.
! 
! This program is free software; you can redistribute it and/or modify it
! under the terms of the GNU Lesser General Public License (as published by
! the Free Software Foundation) version 2.1 dated February 1999.
! 
! This program is distributed in the hope that it will be useful, but
! WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
! conditions of the GNU Lesser General Public License for more details.
! 
! You should have recieved a copy of the GNU Lesser General Public License
! along with this program; if not, write to the Free Software Foundation,
! Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
! 
! WARNING: Automatically generated; changes will be lost
! 
! 


! 
! Symbol "sidl.rmi.Call" (version 0.9.15)
! 
!  
! This interface is implemented by the Server side deserializer.
! Deserializes method arguments in preperation for the method
! call.
! 


#include "sidl_io_Deserializer_fAbbrev.h"
#include "sidl_ClassInfo_fAbbrev.h"
#include "sidl_rmi_Call_fAbbrev.h"
#include "sidl_io_Serializable_fAbbrev.h"
#include "sidl_BaseInterface_fAbbrev.h"
#include "sidl_RuntimeException_fAbbrev.h"
#include "sidl_BaseException_fAbbrev.h"

module sidl_rmi_Call

  use sidl
  use sidl_io_Deserializer_type
  use sidl_ClassInfo_type
  use sidl_rmi_Call_type
  use sidl_io_Serializable_type
  use sidl_BaseInterface_type
  use sidl_RuntimeException_type
  use sidl_BaseException_type
  use sidl_rmi_Call_type
  use sidl_rmi_Return_type
  use sidl_rmi_Ticket_type
  use sidl_char_array
  use sidl_string_array
  use sidl_double_array
  use sidl_opaque_array
  use sidl_long_array
  use sidl_int_array
  use sidl_bool_array
  use sidl_float_array
  use sidl_fcomplex_array
  use sidl_array_array
  use sidl_dcomplex_array

  private :: cast_0, cast_1, cast_2, cast_3
  interface cast
    module procedure cast_0, cast_1, cast_2, cast_3
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
    private :: unpackBool_s


  interface unpackBool
    module procedure unpackBool_s
  end interface
    private :: unpackChar_s


  interface unpackChar
    module procedure unpackChar_s
  end interface
    private :: unpackInt_s


  interface unpackInt
    module procedure unpackInt_s
  end interface
    private :: unpackLong_s


  interface unpackLong
    module procedure unpackLong_s
  end interface
    private :: unpackOpaque_s


  interface unpackOpaque
    module procedure unpackOpaque_s
  end interface
    private :: unpackFloat_s


  interface unpackFloat
    module procedure unpackFloat_s
  end interface
    private :: unpackDouble_s


  interface unpackDouble
    module procedure unpackDouble_s
  end interface
    private :: unpackFcomplex_s


  interface unpackFcomplex
    module procedure unpackFcomplex_s
  end interface
    private :: unpackDcomplex_s


  interface unpackDcomplex
    module procedure unpackDcomplex_s
  end interface
    private :: unpackString_s


  interface unpackString
    module procedure unpackString_s
  end interface
    private :: unpackSerializable_s


  interface unpackSerializable
    module procedure unpackSerializable_s
  end interface
    private :: unpackBoolArray_s


  interface unpackBoolArray
    module procedure unpackBoolArray_s
  end interface
    private :: unpackCharArray_s


  interface unpackCharArray
    module procedure unpackCharArray_s
  end interface
    private :: unpackIntArray_s


  interface unpackIntArray
    module procedure unpackIntArray_s
  end interface
    private :: unpackLongArray_s


  interface unpackLongArray
    module procedure unpackLongArray_s
  end interface
    private :: unpackOpaqueArray_s


  interface unpackOpaqueArray
    module procedure unpackOpaqueArray_s
  end interface
    private :: unpackFloatArray_s


  interface unpackFloatArray
    module procedure unpackFloatArray_s
  end interface
    private :: unpackDoubleArray_s


  interface unpackDoubleArray
    module procedure unpackDoubleArray_s
  end interface
    private :: unpackFcomplexArray_s


  interface unpackFcomplexArray
    module procedure unpackFcomplexArray_s
  end interface
    private :: unpackDcomplexArray_s


  interface unpackDcomplexArray
    module procedure unpackDcomplexArray_s
  end interface
    private :: unpackStringArray_s


  interface unpackStringArray
    module procedure unpackStringArray_s
  end interface
    private :: unpackGenericArray_s


  interface unpackGenericArray
    module procedure unpackGenericArray_s
  end interface
    private :: unpackSerializableArray_s


  interface unpackSerializableArray
    module procedure unpackSerializableArray_s
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



  recursive subroutine rConnect_s(self, url, exception)
    implicit none
    !  out sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(out) :: self
    !  in string url
    character (len=*) , intent(in) :: url
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_rConnect_m
    call sidl_rmi_Call_rConnect_m(self, url, exception)

  end subroutine rConnect_s


  recursive subroutine addRef_s(self, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_addRef_m
    call sidl_rmi_Call_addRef_m(self, exception)

  end subroutine addRef_s


  recursive subroutine deleteRef_s(self, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_deleteRef_m
    call sidl_rmi_Call_deleteRef_m(self, exception)

  end subroutine deleteRef_s


  recursive subroutine isSame_s(self, iobj, retval, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in sidl.BaseInterface iobj
    type(sidl_BaseInterface_t) , intent(in) :: iobj
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_isSame_m
    call sidl_rmi_Call_isSame_m(self, iobj, retval, exception)

  end subroutine isSame_s


  recursive subroutine isType_s(self, name, retval, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string name
    character (len=*) , intent(in) :: name
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_isType_m
    call sidl_rmi_Call_isType_m(self, name, retval, exception)

  end subroutine isType_s


  recursive subroutine getClassInfo_s(self, retval, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  out sidl.ClassInfo retval
    type(sidl_ClassInfo_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_getClassInfo_m
    call sidl_rmi_Call_getClassInfo_m(self, retval, exception)

  end subroutine getClassInfo_s


  recursive subroutine unpackBool_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out bool value
    logical , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackBool_m
    call sidl_rmi_Call_unpackBool_m(self, key, value, exception)

  end subroutine unpackBool_s


  recursive subroutine unpackChar_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out char value
    character (len=1) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackChar_m
    call sidl_rmi_Call_unpackChar_m(self, key, value, exception)

  end subroutine unpackChar_s


  recursive subroutine unpackInt_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out int value
    integer (kind=sidl_int) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackInt_m
    call sidl_rmi_Call_unpackInt_m(self, key, value, exception)

  end subroutine unpackInt_s


  recursive subroutine unpackLong_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out long value
    integer (kind=sidl_long) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackLong_m
    call sidl_rmi_Call_unpackLong_m(self, key, value, exception)

  end subroutine unpackLong_s


  recursive subroutine unpackOpaque_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out opaque value
    integer (kind=sidl_opaque) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackOpaque_m
    call sidl_rmi_Call_unpackOpaque_m(self, key, value, exception)

  end subroutine unpackOpaque_s


  recursive subroutine unpackFloat_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out float value
    real (kind=sidl_float) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackFloat_m
    call sidl_rmi_Call_unpackFloat_m(self, key, value, exception)

  end subroutine unpackFloat_s


  recursive subroutine unpackDouble_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out double value
    real (kind=sidl_double) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackDouble_m
    call sidl_rmi_Call_unpackDouble_m(self, key, value, exception)

  end subroutine unpackDouble_s


  recursive subroutine unpackFcomplex_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out fcomplex value
    complex (kind=sidl_fcomplex) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackFcomplex_m
    call sidl_rmi_Call_unpackFcomplex_m(self, key, value, exception)

  end subroutine unpackFcomplex_s


  recursive subroutine unpackDcomplex_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out dcomplex value
    complex (kind=sidl_dcomplex) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackDcomplex_m
    call sidl_rmi_Call_unpackDcomplex_m(self, key, value, exception)

  end subroutine unpackDcomplex_s


  recursive subroutine unpackString_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out string value
    character (len=*) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackString_m
    call sidl_rmi_Call_unpackString_m(self, key, value, exception)

  end subroutine unpackString_s


  recursive subroutine unpackSerializable_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out sidl.io.Serializable value
    type(sidl_io_Serializable_t) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackSerializable_m
    call sidl_rmi_Call_unpackSerializable_m(self, key, value, exception)

  end subroutine unpackSerializable_s


  recursive subroutine unpackBoolArray_s(self, key, value, ordering, dimen,    &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<bool> value
    type(sidl_bool_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackBoolArray_m
    call sidl_rmi_Call_unpackBoolArray_m(self, key, value, ordering, dimen,    &
      isRarray, exception)

  end subroutine unpackBoolArray_s


  recursive subroutine unpackCharArray_s(self, key, value, ordering, dimen,    &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<char> value
    type(sidl_char_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackCharArray_m
    call sidl_rmi_Call_unpackCharArray_m(self, key, value, ordering, dimen,    &
      isRarray, exception)

  end subroutine unpackCharArray_s


  recursive subroutine unpackIntArray_s(self, key, value, ordering, dimen,     &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<int> value
    type(sidl_int_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackIntArray_m
    call sidl_rmi_Call_unpackIntArray_m(self, key, value, ordering, dimen,     &
      isRarray, exception)

  end subroutine unpackIntArray_s


  recursive subroutine unpackLongArray_s(self, key, value, ordering, dimen,    &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<long> value
    type(sidl_long_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackLongArray_m
    call sidl_rmi_Call_unpackLongArray_m(self, key, value, ordering, dimen,    &
      isRarray, exception)

  end subroutine unpackLongArray_s


  recursive subroutine unpackOpaqueArray_s(self, key, value, ordering, dimen,  &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<opaque> value
    type(sidl_opaque_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackOpaqueArray_m
    call sidl_rmi_Call_unpackOpaqueArray_m(self, key, value, ordering, dimen,  &
      isRarray, exception)

  end subroutine unpackOpaqueArray_s


  recursive subroutine unpackFloatArray_s(self, key, value, ordering, dimen,   &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<float> value
    type(sidl_float_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackFloatArray_m
    call sidl_rmi_Call_unpackFloatArray_m(self, key, value, ordering, dimen,   &
      isRarray, exception)

  end subroutine unpackFloatArray_s


  recursive subroutine unpackDoubleArray_s(self, key, value, ordering, dimen,  &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<double> value
    type(sidl_double_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackDoubleArray_m
    call sidl_rmi_Call_unpackDoubleArray_m(self, key, value, ordering, dimen,  &
      isRarray, exception)

  end subroutine unpackDoubleArray_s


  recursive subroutine unpackFcomplexArray_s(self, key, value, ordering,       &
    dimen, isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<fcomplex> value
    type(sidl_fcomplex_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackFcomplexArray_m
    call sidl_rmi_Call_unpackFcomplexArray_m(self, key, value, ordering,       &
      dimen, isRarray, exception)

  end subroutine unpackFcomplexArray_s


  recursive subroutine unpackDcomplexArray_s(self, key, value, ordering,       &
    dimen, isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<dcomplex> value
    type(sidl_dcomplex_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackDcomplexArray_m
    call sidl_rmi_Call_unpackDcomplexArray_m(self, key, value, ordering,       &
      dimen, isRarray, exception)

  end subroutine unpackDcomplexArray_s


  recursive subroutine unpackStringArray_s(self, key, value, ordering, dimen,  &
    isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<string> value
    type(sidl_string_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackStringArray_m
    call sidl_rmi_Call_unpackStringArray_m(self, key, value, ordering, dimen,  &
      isRarray, exception)

  end subroutine unpackStringArray_s


  recursive subroutine unpackGenericArray_s(self, key, value, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<> value
    type(sidl__array) , intent(out) :: value
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackGenericArray_m
    call sidl_rmi_Call_unpackGenericArray_m(self, key, value, exception)

  end subroutine unpackGenericArray_s


  recursive subroutine unpackSerializableArray_s(self, key, value, ordering,   &
    dimen, isRarray, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string key
    character (len=*) , intent(in) :: key
    !  out array<sidl.io.Serializable> value
    type(sidl_io_Serializable_1d) , intent(out) :: value
    !  in int ordering
    integer (kind=sidl_int) , intent(in) :: ordering
    !  in int dimen
    integer (kind=sidl_int) , intent(in) :: dimen
    !  in bool isRarray
    logical , intent(in) :: isRarray
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call_unpackSerializableArray_m
    call sidl_rmi_Call_unpackSerializableArray_m(self, key, value, ordering,   &
      dimen, isRarray, exception)

  end subroutine unpackSerializableArray_s

  ! 
  ! Static function to cast from sidl.rmi.Call
  ! to sidl.BaseInterface.
  ! 

  subroutine cast_0(oldType, newType, exception)
    implicit none
    type(sidl_rmi_Call_t), intent(in) :: oldType
    type(sidl_BaseInterface_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_BaseInterface__cast_m

    call sidl_BaseInterface__cast_m(oldType, newType, exception)
  end subroutine cast_0

  ! 
  ! Static function to cast from sidl.BaseInterface
  ! to sidl.rmi.Call.
  ! 

  subroutine cast_1(oldType, newType, exception)
    implicit none
    type(sidl_BaseInterface_t), intent(in) :: oldType
    type(sidl_rmi_Call_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_rmi_Call__cast_m

    call sidl_rmi_Call__cast_m(oldType, newType, exception)
  end subroutine cast_1

  ! 
  ! Static function to cast from sidl.rmi.Call
  ! to sidl.io.Deserializer.
  ! 

  subroutine cast_2(oldType, newType, exception)
    implicit none
    type(sidl_rmi_Call_t), intent(in) :: oldType
    type(sidl_io_Deserializer_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_io_Deserializer__cast_m

    call sidl_io_Deserializer__cast_m(oldType, newType, exception)
  end subroutine cast_2

  ! 
  ! Static function to cast from sidl.io.Deserializer
  ! to sidl.rmi.Call.
  ! 

  subroutine cast_3(oldType, newType, exception)
    implicit none
    type(sidl_io_Deserializer_t), intent(in) :: oldType
    type(sidl_rmi_Call_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_rmi_Call__cast_m

    call sidl_rmi_Call__cast_m(oldType, newType, exception)
  end subroutine cast_3


  recursive subroutine exec_s(self, methodName, inArgs, outArgs, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in string methodName
    character (len=*) , intent(in) :: methodName
    !  in sidl.rmi.Call inArgs
    type(sidl_rmi_Call_t) , intent(in) :: inArgs
    !  in sidl.rmi.Return outArgs
    type(sidl_rmi_Return_t) , intent(in) :: outArgs
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call__exec_m
    call sidl_rmi_Call__exec_m(self, methodName, inArgs, outArgs, exception)

  end subroutine exec_s

  recursive subroutine getURL_s(self, retval, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  out string retval
    character (len=*) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call__getURL_m
    call sidl_rmi_Call__getURL_m(self, retval, exception)

  end subroutine getURL_s

  recursive subroutine isRemote_s(self, retval, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call__isRemote_m
    call sidl_rmi_Call__isRemote_m(self, retval, exception)

  end subroutine isRemote_s

  recursive subroutine isLocal_s(self, retval, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call__isLocal_m
    call sidl_rmi_Call__isLocal_m(self, retval, exception)

  end subroutine isLocal_s

  recursive subroutine set_hooks_s(self, on, exception)
    implicit none
    !  in sidl.rmi.Call self
    type(sidl_rmi_Call_t) , intent(in) :: self
    !  in bool on
    logical , intent(in) :: on
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_rmi_Call__set_hooks_m
    call sidl_rmi_Call__set_hooks_m(self, on, exception)

  end subroutine set_hooks_s
  logical function is_null_s(ext)
    type(sidl_rmi_Call_t), intent(in) :: ext
    is_null_s = (ext%d_ior .eq. 0)
  end function is_null_s

  logical function not_null_s(ext)
    type(sidl_rmi_Call_t), intent(in) :: ext
    not_null_s = (ext%d_ior .ne. 0)
  end function not_null_s

  subroutine set_null_s(ext)
    type(sidl_rmi_Call_t), intent(out) :: ext
    ext%d_ior = 0
  end subroutine set_null_s


end module sidl_rmi_Call

module sidl_rmi_Call_array
  use sidl
  use sidl_rmi_Call_type
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
    castsidl_rmi_Call1dToGeneric_p, &
    castsidl_rmi_Call2dToGeneric_p, &
    castsidl_rmi_Call3dToGeneric_p, &
    castsidl_rmi_Call4dToGeneric_p, &
    castsidl_rmi_Call5dToGeneric_p, &
    castsidl_rmi_Call6dToGeneric_p, &
    castsidl_rmi_Call7dToGeneric_p
interface cast
  module procedure &
    castsidl_rmi_Call1dToGeneric_p, &
    castsidl_rmi_Call2dToGeneric_p, &
    castsidl_rmi_Call3dToGeneric_p, &
    castsidl_rmi_Call4dToGeneric_p, &
    castsidl_rmi_Call5dToGeneric_p, &
    castsidl_rmi_Call6dToGeneric_p, &
    castsidl_rmi_Call7dToGeneric_p
end interface


contains


  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(sidl_rmi_Call_1d), intent(out) :: array
    external Ca_ary_createColfj08hmklxyrlh_m
    call Ca_ary_createColfj08hmklxyrlh_m(1, lower, upper, array)
  end subroutine createCol1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(sidl_rmi_Call_1d), intent(out) :: array
    external Ca_ary_createRowmndjuf5ps18ki_m
    call Ca_ary_createRowmndjuf5ps18ki_m(1, lower, upper, array)
  end subroutine createRow1_p

  subroutine create1d1_p(len, array)
    integer (kind=sidl_int), intent(in) :: len
    type(sidl_rmi_Call_1d), intent(out) :: array
    external sidl_rmi_Call__array_create1d_m
    call sidl_rmi_Call__array_create1d_m(len, array)
  end subroutine create1d1_p

  subroutine copy1_p(src, dest)
    type(sidl_rmi_Call_1d), intent(in) :: src
    type(sidl_rmi_Call_1d), intent(in) :: dest
    external sidl_rmi_Call__array_copy_m
    call sidl_rmi_Call__array_copy_m(src, dest)
  end subroutine copy1_p

  subroutine ensure1_p(src, dim, ordering, result)
    type(sidl_rmi_Call_1d), intent(in)  :: src
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_rmi_Call__array_ensure_m
    call sidl_rmi_Call__array_ensure_m(src, 1, ordering, result)
  end subroutine ensure1_p

  subroutine slice11_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_1d), intent(in)  :: src
    integer (kind=sidl_int), dimension(1), intent(in) :: numElem
    integer (kind=sidl_int), dimension(1), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 1, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice11_p

  subroutine getg1_p(array, index, value)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get_m
    call sidl_rmi_Call__array_get_m(array, index, value)
  end subroutine getg1_p

  subroutine setg1_p(array, index, value)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set_m
    call sidl_rmi_Call__array_set_m(array, index, value)
  end subroutine setg1_p

  subroutine get1_p(array, &
      i1, &
      value)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get1_m
    call sidl_rmi_Call__array_get1_m(array, &
      i1, &
      value)
  end subroutine get1_p

  subroutine set1_p(array, &
      i1, &
      value)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set1_m
    call sidl_rmi_Call__array_set1_m(array, &
      i1, &
      value)
  end subroutine set1_p

  subroutine smartCopy1_p(src, dest)
    type(sidl_rmi_Call_1d), intent(in) :: src
    type(sidl_rmi_Call_1d), intent(out) :: dest
    integer(sidl_int) :: dim
    external Ca_ary_smartCopyuxys5ilkcx0hr_m
    dim = 1
    call Ca_ary_smartCopyuxys5ilkcx0hr_m(src, 1, dest)
  end subroutine smartCopy1_p

  logical function  isColumnOrder1_p(array)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    external ary_isColumnOrderbmi6s8sbzc4u_m
    call ary_isColumnOrderbmi6s8sbzc4u_m(array, isColumnOrder1_p)
  end function isColumnOrder1_p

  logical function  isRowOrder1_p(array)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    external C_ary_isRowOrderhfaiet9x2d5_o_m
    call C_ary_isRowOrderhfaiet9x2d5_o_m(array, isRowOrder1_p)
  end function isRowOrder1_p

  integer (kind=sidl_int) function  dimen1_p(array)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    external sidl_rmi_Call__array_dimen_m
    call sidl_rmi_Call__array_dimen_m(array, dimen1_p)
  end function dimen1_p

  integer (kind=sidl_int) function  stride1_p(array, index)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_stride_m
    call sidl_rmi_Call__array_stride_m(array, index, stride1_p)
  end function stride1_p

  integer (kind=sidl_int) function  lower1_p(array, index)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_lower_m
    call sidl_rmi_Call__array_lower_m(array, index, lower1_p)
  end function lower1_p

  integer (kind=sidl_int) function  upper1_p(array, index)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_upper_m
    call sidl_rmi_Call__array_upper_m(array, index, upper1_p)
  end function upper1_p

  integer (kind=sidl_int) function  length1_p(array, index)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_length_m
    call sidl_rmi_Call__array_length_m(array, index, length1_p)
  end function length1_p

  subroutine  addRef1_p(array)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    external sidl_rmi_Call__array_addRef_m
    call sidl_rmi_Call__array_addRef_m(array)
  end subroutine addRef1_p

  subroutine  deleteRef1_p(array)
    type(sidl_rmi_Call_1d), intent(in)  :: array
    external Ca_ary_deleteRefrpoidx9is5tpw_m
    call Ca_ary_deleteRefrpoidx9is5tpw_m(array)
  end subroutine deleteRef1_p

  logical function is_null1_p(array)
    type(sidl_rmi_Call_1d), intent(in) :: array
    is_null1_p = (array%d_array .eq. 0)
  end function is_null1_p

  logical function not_null1_p(array)
    type(sidl_rmi_Call_1d), intent(in) :: array
    not_null1_p = (array%d_array .ne. 0)
  end function not_null1_p

  subroutine set_null1_p(array)
    type(sidl_rmi_Call_1d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(sidl_rmi_Call_2d), intent(out) :: array
    external Ca_ary_createColfj08hmklxyrlh_m
    call Ca_ary_createColfj08hmklxyrlh_m(2, lower, upper, array)
  end subroutine createCol2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(sidl_rmi_Call_2d), intent(out) :: array
    external Ca_ary_createRowmndjuf5ps18ki_m
    call Ca_ary_createRowmndjuf5ps18ki_m(2, lower, upper, array)
  end subroutine createRow2_p

  subroutine create2dCol2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(sidl_rmi_Call_2d), intent(out) :: array
    external ary_create2dCol2slyzsg9o19lq0_m
    call ary_create2dCol2slyzsg9o19lq0_m(m, n, array)
  end subroutine create2dCol2_p

  subroutine create2dRow2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(sidl_rmi_Call_2d), intent(out) :: array
    external ary_create2dRowc68ysuyq114gdy_m
    call ary_create2dRowc68ysuyq114gdy_m(m, n, array)
  end subroutine create2dRow2_p

  subroutine copy2_p(src, dest)
    type(sidl_rmi_Call_2d), intent(in) :: src
    type(sidl_rmi_Call_2d), intent(in) :: dest
    external sidl_rmi_Call__array_copy_m
    call sidl_rmi_Call__array_copy_m(src, dest)
  end subroutine copy2_p

  subroutine ensure2_p(src, dim, ordering, result)
    type(sidl_rmi_Call_2d), intent(in)  :: src
    type(sidl_rmi_Call_2d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_rmi_Call__array_ensure_m
    call sidl_rmi_Call__array_ensure_m(src, 2, ordering, result)
  end subroutine ensure2_p

  subroutine slice12_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 1, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice12_p

  subroutine slice22_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 2, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice22_p

  subroutine getg2_p(array, index, value)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get_m
    call sidl_rmi_Call__array_get_m(array, index, value)
  end subroutine getg2_p

  subroutine setg2_p(array, index, value)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set_m
    call sidl_rmi_Call__array_set_m(array, index, value)
  end subroutine setg2_p

  subroutine get2_p(array, &
      i1, &
      i2, &
      value)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get2_m
    call sidl_rmi_Call__array_get2_m(array, &
      i1, &
      i2, &
      value)
  end subroutine get2_p

  subroutine set2_p(array, &
      i1, &
      i2, &
      value)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set2_m
    call sidl_rmi_Call__array_set2_m(array, &
      i1, &
      i2, &
      value)
  end subroutine set2_p

  subroutine smartCopy2_p(src, dest)
    type(sidl_rmi_Call_2d), intent(in) :: src
    type(sidl_rmi_Call_2d), intent(out) :: dest
    integer(sidl_int) :: dim
    external Ca_ary_smartCopyuxys5ilkcx0hr_m
    dim = 2
    call Ca_ary_smartCopyuxys5ilkcx0hr_m(src, 2, dest)
  end subroutine smartCopy2_p

  logical function  isColumnOrder2_p(array)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    external ary_isColumnOrderbmi6s8sbzc4u_m
    call ary_isColumnOrderbmi6s8sbzc4u_m(array, isColumnOrder2_p)
  end function isColumnOrder2_p

  logical function  isRowOrder2_p(array)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    external C_ary_isRowOrderhfaiet9x2d5_o_m
    call C_ary_isRowOrderhfaiet9x2d5_o_m(array, isRowOrder2_p)
  end function isRowOrder2_p

  integer (kind=sidl_int) function  dimen2_p(array)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    external sidl_rmi_Call__array_dimen_m
    call sidl_rmi_Call__array_dimen_m(array, dimen2_p)
  end function dimen2_p

  integer (kind=sidl_int) function  stride2_p(array, index)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_stride_m
    call sidl_rmi_Call__array_stride_m(array, index, stride2_p)
  end function stride2_p

  integer (kind=sidl_int) function  lower2_p(array, index)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_lower_m
    call sidl_rmi_Call__array_lower_m(array, index, lower2_p)
  end function lower2_p

  integer (kind=sidl_int) function  upper2_p(array, index)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_upper_m
    call sidl_rmi_Call__array_upper_m(array, index, upper2_p)
  end function upper2_p

  integer (kind=sidl_int) function  length2_p(array, index)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_length_m
    call sidl_rmi_Call__array_length_m(array, index, length2_p)
  end function length2_p

  subroutine  addRef2_p(array)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    external sidl_rmi_Call__array_addRef_m
    call sidl_rmi_Call__array_addRef_m(array)
  end subroutine addRef2_p

  subroutine  deleteRef2_p(array)
    type(sidl_rmi_Call_2d), intent(in)  :: array
    external Ca_ary_deleteRefrpoidx9is5tpw_m
    call Ca_ary_deleteRefrpoidx9is5tpw_m(array)
  end subroutine deleteRef2_p

  logical function is_null2_p(array)
    type(sidl_rmi_Call_2d), intent(in) :: array
    is_null2_p = (array%d_array .eq. 0)
  end function is_null2_p

  logical function not_null2_p(array)
    type(sidl_rmi_Call_2d), intent(in) :: array
    not_null2_p = (array%d_array .ne. 0)
  end function not_null2_p

  subroutine set_null2_p(array)
    type(sidl_rmi_Call_2d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(sidl_rmi_Call_3d), intent(out) :: array
    external Ca_ary_createColfj08hmklxyrlh_m
    call Ca_ary_createColfj08hmklxyrlh_m(3, lower, upper, array)
  end subroutine createCol3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(sidl_rmi_Call_3d), intent(out) :: array
    external Ca_ary_createRowmndjuf5ps18ki_m
    call Ca_ary_createRowmndjuf5ps18ki_m(3, lower, upper, array)
  end subroutine createRow3_p

  subroutine copy3_p(src, dest)
    type(sidl_rmi_Call_3d), intent(in) :: src
    type(sidl_rmi_Call_3d), intent(in) :: dest
    external sidl_rmi_Call__array_copy_m
    call sidl_rmi_Call__array_copy_m(src, dest)
  end subroutine copy3_p

  subroutine ensure3_p(src, dim, ordering, result)
    type(sidl_rmi_Call_3d), intent(in)  :: src
    type(sidl_rmi_Call_3d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_rmi_Call__array_ensure_m
    call sidl_rmi_Call__array_ensure_m(src, 3, ordering, result)
  end subroutine ensure3_p

  subroutine slice13_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 1, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice13_p

  subroutine slice23_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 2, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice23_p

  subroutine slice33_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 3, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice33_p

  subroutine getg3_p(array, index, value)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get_m
    call sidl_rmi_Call__array_get_m(array, index, value)
  end subroutine getg3_p

  subroutine setg3_p(array, index, value)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set_m
    call sidl_rmi_Call__array_set_m(array, index, value)
  end subroutine setg3_p

  subroutine get3_p(array, &
      i1, &
      i2, &
      i3, &
      value)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get3_m
    call sidl_rmi_Call__array_get3_m(array, &
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
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set3_m
    call sidl_rmi_Call__array_set3_m(array, &
      i1, &
      i2, &
      i3, &
      value)
  end subroutine set3_p

  subroutine smartCopy3_p(src, dest)
    type(sidl_rmi_Call_3d), intent(in) :: src
    type(sidl_rmi_Call_3d), intent(out) :: dest
    integer(sidl_int) :: dim
    external Ca_ary_smartCopyuxys5ilkcx0hr_m
    dim = 3
    call Ca_ary_smartCopyuxys5ilkcx0hr_m(src, 3, dest)
  end subroutine smartCopy3_p

  logical function  isColumnOrder3_p(array)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    external ary_isColumnOrderbmi6s8sbzc4u_m
    call ary_isColumnOrderbmi6s8sbzc4u_m(array, isColumnOrder3_p)
  end function isColumnOrder3_p

  logical function  isRowOrder3_p(array)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    external C_ary_isRowOrderhfaiet9x2d5_o_m
    call C_ary_isRowOrderhfaiet9x2d5_o_m(array, isRowOrder3_p)
  end function isRowOrder3_p

  integer (kind=sidl_int) function  dimen3_p(array)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    external sidl_rmi_Call__array_dimen_m
    call sidl_rmi_Call__array_dimen_m(array, dimen3_p)
  end function dimen3_p

  integer (kind=sidl_int) function  stride3_p(array, index)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_stride_m
    call sidl_rmi_Call__array_stride_m(array, index, stride3_p)
  end function stride3_p

  integer (kind=sidl_int) function  lower3_p(array, index)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_lower_m
    call sidl_rmi_Call__array_lower_m(array, index, lower3_p)
  end function lower3_p

  integer (kind=sidl_int) function  upper3_p(array, index)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_upper_m
    call sidl_rmi_Call__array_upper_m(array, index, upper3_p)
  end function upper3_p

  integer (kind=sidl_int) function  length3_p(array, index)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_length_m
    call sidl_rmi_Call__array_length_m(array, index, length3_p)
  end function length3_p

  subroutine  addRef3_p(array)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    external sidl_rmi_Call__array_addRef_m
    call sidl_rmi_Call__array_addRef_m(array)
  end subroutine addRef3_p

  subroutine  deleteRef3_p(array)
    type(sidl_rmi_Call_3d), intent(in)  :: array
    external Ca_ary_deleteRefrpoidx9is5tpw_m
    call Ca_ary_deleteRefrpoidx9is5tpw_m(array)
  end subroutine deleteRef3_p

  logical function is_null3_p(array)
    type(sidl_rmi_Call_3d), intent(in) :: array
    is_null3_p = (array%d_array .eq. 0)
  end function is_null3_p

  logical function not_null3_p(array)
    type(sidl_rmi_Call_3d), intent(in) :: array
    not_null3_p = (array%d_array .ne. 0)
  end function not_null3_p

  subroutine set_null3_p(array)
    type(sidl_rmi_Call_3d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(sidl_rmi_Call_4d), intent(out) :: array
    external Ca_ary_createColfj08hmklxyrlh_m
    call Ca_ary_createColfj08hmklxyrlh_m(4, lower, upper, array)
  end subroutine createCol4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(sidl_rmi_Call_4d), intent(out) :: array
    external Ca_ary_createRowmndjuf5ps18ki_m
    call Ca_ary_createRowmndjuf5ps18ki_m(4, lower, upper, array)
  end subroutine createRow4_p

  subroutine copy4_p(src, dest)
    type(sidl_rmi_Call_4d), intent(in) :: src
    type(sidl_rmi_Call_4d), intent(in) :: dest
    external sidl_rmi_Call__array_copy_m
    call sidl_rmi_Call__array_copy_m(src, dest)
  end subroutine copy4_p

  subroutine ensure4_p(src, dim, ordering, result)
    type(sidl_rmi_Call_4d), intent(in)  :: src
    type(sidl_rmi_Call_4d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_rmi_Call__array_ensure_m
    call sidl_rmi_Call__array_ensure_m(src, 4, ordering, result)
  end subroutine ensure4_p

  subroutine slice14_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 1, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice14_p

  subroutine slice24_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 2, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice24_p

  subroutine slice34_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 3, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice34_p

  subroutine slice44_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 4, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice44_p

  subroutine getg4_p(array, index, value)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get_m
    call sidl_rmi_Call__array_get_m(array, index, value)
  end subroutine getg4_p

  subroutine setg4_p(array, index, value)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set_m
    call sidl_rmi_Call__array_set_m(array, index, value)
  end subroutine setg4_p

  subroutine get4_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get4_m
    call sidl_rmi_Call__array_get4_m(array, &
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
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set4_m
    call sidl_rmi_Call__array_set4_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
  end subroutine set4_p

  subroutine smartCopy4_p(src, dest)
    type(sidl_rmi_Call_4d), intent(in) :: src
    type(sidl_rmi_Call_4d), intent(out) :: dest
    integer(sidl_int) :: dim
    external Ca_ary_smartCopyuxys5ilkcx0hr_m
    dim = 4
    call Ca_ary_smartCopyuxys5ilkcx0hr_m(src, 4, dest)
  end subroutine smartCopy4_p

  logical function  isColumnOrder4_p(array)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    external ary_isColumnOrderbmi6s8sbzc4u_m
    call ary_isColumnOrderbmi6s8sbzc4u_m(array, isColumnOrder4_p)
  end function isColumnOrder4_p

  logical function  isRowOrder4_p(array)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    external C_ary_isRowOrderhfaiet9x2d5_o_m
    call C_ary_isRowOrderhfaiet9x2d5_o_m(array, isRowOrder4_p)
  end function isRowOrder4_p

  integer (kind=sidl_int) function  dimen4_p(array)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    external sidl_rmi_Call__array_dimen_m
    call sidl_rmi_Call__array_dimen_m(array, dimen4_p)
  end function dimen4_p

  integer (kind=sidl_int) function  stride4_p(array, index)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_stride_m
    call sidl_rmi_Call__array_stride_m(array, index, stride4_p)
  end function stride4_p

  integer (kind=sidl_int) function  lower4_p(array, index)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_lower_m
    call sidl_rmi_Call__array_lower_m(array, index, lower4_p)
  end function lower4_p

  integer (kind=sidl_int) function  upper4_p(array, index)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_upper_m
    call sidl_rmi_Call__array_upper_m(array, index, upper4_p)
  end function upper4_p

  integer (kind=sidl_int) function  length4_p(array, index)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_length_m
    call sidl_rmi_Call__array_length_m(array, index, length4_p)
  end function length4_p

  subroutine  addRef4_p(array)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    external sidl_rmi_Call__array_addRef_m
    call sidl_rmi_Call__array_addRef_m(array)
  end subroutine addRef4_p

  subroutine  deleteRef4_p(array)
    type(sidl_rmi_Call_4d), intent(in)  :: array
    external Ca_ary_deleteRefrpoidx9is5tpw_m
    call Ca_ary_deleteRefrpoidx9is5tpw_m(array)
  end subroutine deleteRef4_p

  logical function is_null4_p(array)
    type(sidl_rmi_Call_4d), intent(in) :: array
    is_null4_p = (array%d_array .eq. 0)
  end function is_null4_p

  logical function not_null4_p(array)
    type(sidl_rmi_Call_4d), intent(in) :: array
    not_null4_p = (array%d_array .ne. 0)
  end function not_null4_p

  subroutine set_null4_p(array)
    type(sidl_rmi_Call_4d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(sidl_rmi_Call_5d), intent(out) :: array
    external Ca_ary_createColfj08hmklxyrlh_m
    call Ca_ary_createColfj08hmklxyrlh_m(5, lower, upper, array)
  end subroutine createCol5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(sidl_rmi_Call_5d), intent(out) :: array
    external Ca_ary_createRowmndjuf5ps18ki_m
    call Ca_ary_createRowmndjuf5ps18ki_m(5, lower, upper, array)
  end subroutine createRow5_p

  subroutine copy5_p(src, dest)
    type(sidl_rmi_Call_5d), intent(in) :: src
    type(sidl_rmi_Call_5d), intent(in) :: dest
    external sidl_rmi_Call__array_copy_m
    call sidl_rmi_Call__array_copy_m(src, dest)
  end subroutine copy5_p

  subroutine ensure5_p(src, dim, ordering, result)
    type(sidl_rmi_Call_5d), intent(in)  :: src
    type(sidl_rmi_Call_5d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_rmi_Call__array_ensure_m
    call sidl_rmi_Call__array_ensure_m(src, 5, ordering, result)
  end subroutine ensure5_p

  subroutine slice15_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 1, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice15_p

  subroutine slice25_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 2, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice25_p

  subroutine slice35_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 3, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice35_p

  subroutine slice45_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 4, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice45_p

  subroutine slice55_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 5, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice55_p

  subroutine getg5_p(array, index, value)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get_m
    call sidl_rmi_Call__array_get_m(array, index, value)
  end subroutine getg5_p

  subroutine setg5_p(array, index, value)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set_m
    call sidl_rmi_Call__array_set_m(array, index, value)
  end subroutine setg5_p

  subroutine get5_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get5_m
    call sidl_rmi_Call__array_get5_m(array, &
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
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set5_m
    call sidl_rmi_Call__array_set5_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
  end subroutine set5_p

  subroutine smartCopy5_p(src, dest)
    type(sidl_rmi_Call_5d), intent(in) :: src
    type(sidl_rmi_Call_5d), intent(out) :: dest
    integer(sidl_int) :: dim
    external Ca_ary_smartCopyuxys5ilkcx0hr_m
    dim = 5
    call Ca_ary_smartCopyuxys5ilkcx0hr_m(src, 5, dest)
  end subroutine smartCopy5_p

  logical function  isColumnOrder5_p(array)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    external ary_isColumnOrderbmi6s8sbzc4u_m
    call ary_isColumnOrderbmi6s8sbzc4u_m(array, isColumnOrder5_p)
  end function isColumnOrder5_p

  logical function  isRowOrder5_p(array)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    external C_ary_isRowOrderhfaiet9x2d5_o_m
    call C_ary_isRowOrderhfaiet9x2d5_o_m(array, isRowOrder5_p)
  end function isRowOrder5_p

  integer (kind=sidl_int) function  dimen5_p(array)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    external sidl_rmi_Call__array_dimen_m
    call sidl_rmi_Call__array_dimen_m(array, dimen5_p)
  end function dimen5_p

  integer (kind=sidl_int) function  stride5_p(array, index)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_stride_m
    call sidl_rmi_Call__array_stride_m(array, index, stride5_p)
  end function stride5_p

  integer (kind=sidl_int) function  lower5_p(array, index)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_lower_m
    call sidl_rmi_Call__array_lower_m(array, index, lower5_p)
  end function lower5_p

  integer (kind=sidl_int) function  upper5_p(array, index)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_upper_m
    call sidl_rmi_Call__array_upper_m(array, index, upper5_p)
  end function upper5_p

  integer (kind=sidl_int) function  length5_p(array, index)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_length_m
    call sidl_rmi_Call__array_length_m(array, index, length5_p)
  end function length5_p

  subroutine  addRef5_p(array)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    external sidl_rmi_Call__array_addRef_m
    call sidl_rmi_Call__array_addRef_m(array)
  end subroutine addRef5_p

  subroutine  deleteRef5_p(array)
    type(sidl_rmi_Call_5d), intent(in)  :: array
    external Ca_ary_deleteRefrpoidx9is5tpw_m
    call Ca_ary_deleteRefrpoidx9is5tpw_m(array)
  end subroutine deleteRef5_p

  logical function is_null5_p(array)
    type(sidl_rmi_Call_5d), intent(in) :: array
    is_null5_p = (array%d_array .eq. 0)
  end function is_null5_p

  logical function not_null5_p(array)
    type(sidl_rmi_Call_5d), intent(in) :: array
    not_null5_p = (array%d_array .ne. 0)
  end function not_null5_p

  subroutine set_null5_p(array)
    type(sidl_rmi_Call_5d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(sidl_rmi_Call_6d), intent(out) :: array
    external Ca_ary_createColfj08hmklxyrlh_m
    call Ca_ary_createColfj08hmklxyrlh_m(6, lower, upper, array)
  end subroutine createCol6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(sidl_rmi_Call_6d), intent(out) :: array
    external Ca_ary_createRowmndjuf5ps18ki_m
    call Ca_ary_createRowmndjuf5ps18ki_m(6, lower, upper, array)
  end subroutine createRow6_p

  subroutine copy6_p(src, dest)
    type(sidl_rmi_Call_6d), intent(in) :: src
    type(sidl_rmi_Call_6d), intent(in) :: dest
    external sidl_rmi_Call__array_copy_m
    call sidl_rmi_Call__array_copy_m(src, dest)
  end subroutine copy6_p

  subroutine ensure6_p(src, dim, ordering, result)
    type(sidl_rmi_Call_6d), intent(in)  :: src
    type(sidl_rmi_Call_6d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_rmi_Call__array_ensure_m
    call sidl_rmi_Call__array_ensure_m(src, 6, ordering, result)
  end subroutine ensure6_p

  subroutine slice16_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 1, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice16_p

  subroutine slice26_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 2, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice26_p

  subroutine slice36_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 3, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice36_p

  subroutine slice46_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 4, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice46_p

  subroutine slice56_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 5, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice56_p

  subroutine slice66_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 6, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice66_p

  subroutine getg6_p(array, index, value)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get_m
    call sidl_rmi_Call__array_get_m(array, index, value)
  end subroutine getg6_p

  subroutine setg6_p(array, index, value)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set_m
    call sidl_rmi_Call__array_set_m(array, index, value)
  end subroutine setg6_p

  subroutine get6_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get6_m
    call sidl_rmi_Call__array_get6_m(array, &
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
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set6_m
    call sidl_rmi_Call__array_set6_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
  end subroutine set6_p

  subroutine smartCopy6_p(src, dest)
    type(sidl_rmi_Call_6d), intent(in) :: src
    type(sidl_rmi_Call_6d), intent(out) :: dest
    integer(sidl_int) :: dim
    external Ca_ary_smartCopyuxys5ilkcx0hr_m
    dim = 6
    call Ca_ary_smartCopyuxys5ilkcx0hr_m(src, 6, dest)
  end subroutine smartCopy6_p

  logical function  isColumnOrder6_p(array)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    external ary_isColumnOrderbmi6s8sbzc4u_m
    call ary_isColumnOrderbmi6s8sbzc4u_m(array, isColumnOrder6_p)
  end function isColumnOrder6_p

  logical function  isRowOrder6_p(array)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    external C_ary_isRowOrderhfaiet9x2d5_o_m
    call C_ary_isRowOrderhfaiet9x2d5_o_m(array, isRowOrder6_p)
  end function isRowOrder6_p

  integer (kind=sidl_int) function  dimen6_p(array)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    external sidl_rmi_Call__array_dimen_m
    call sidl_rmi_Call__array_dimen_m(array, dimen6_p)
  end function dimen6_p

  integer (kind=sidl_int) function  stride6_p(array, index)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_stride_m
    call sidl_rmi_Call__array_stride_m(array, index, stride6_p)
  end function stride6_p

  integer (kind=sidl_int) function  lower6_p(array, index)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_lower_m
    call sidl_rmi_Call__array_lower_m(array, index, lower6_p)
  end function lower6_p

  integer (kind=sidl_int) function  upper6_p(array, index)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_upper_m
    call sidl_rmi_Call__array_upper_m(array, index, upper6_p)
  end function upper6_p

  integer (kind=sidl_int) function  length6_p(array, index)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_length_m
    call sidl_rmi_Call__array_length_m(array, index, length6_p)
  end function length6_p

  subroutine  addRef6_p(array)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    external sidl_rmi_Call__array_addRef_m
    call sidl_rmi_Call__array_addRef_m(array)
  end subroutine addRef6_p

  subroutine  deleteRef6_p(array)
    type(sidl_rmi_Call_6d), intent(in)  :: array
    external Ca_ary_deleteRefrpoidx9is5tpw_m
    call Ca_ary_deleteRefrpoidx9is5tpw_m(array)
  end subroutine deleteRef6_p

  logical function is_null6_p(array)
    type(sidl_rmi_Call_6d), intent(in) :: array
    is_null6_p = (array%d_array .eq. 0)
  end function is_null6_p

  logical function not_null6_p(array)
    type(sidl_rmi_Call_6d), intent(in) :: array
    not_null6_p = (array%d_array .ne. 0)
  end function not_null6_p

  subroutine set_null6_p(array)
    type(sidl_rmi_Call_6d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(sidl_rmi_Call_7d), intent(out) :: array
    external Ca_ary_createColfj08hmklxyrlh_m
    call Ca_ary_createColfj08hmklxyrlh_m(7, lower, upper, array)
  end subroutine createCol7_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(sidl_rmi_Call_7d), intent(out) :: array
    external Ca_ary_createRowmndjuf5ps18ki_m
    call Ca_ary_createRowmndjuf5ps18ki_m(7, lower, upper, array)
  end subroutine createRow7_p

  subroutine copy7_p(src, dest)
    type(sidl_rmi_Call_7d), intent(in) :: src
    type(sidl_rmi_Call_7d), intent(in) :: dest
    external sidl_rmi_Call__array_copy_m
    call sidl_rmi_Call__array_copy_m(src, dest)
  end subroutine copy7_p

  subroutine ensure7_p(src, dim, ordering, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    type(sidl_rmi_Call_7d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_rmi_Call__array_ensure_m
    call sidl_rmi_Call__array_ensure_m(src, 7, ordering, result)
  end subroutine ensure7_p

  subroutine slice17_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 1, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice17_p

  subroutine slice27_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 2, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice27_p

  subroutine slice37_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 3, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice37_p

  subroutine slice47_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 4, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice47_p

  subroutine slice57_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 5, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice57_p

  subroutine slice67_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 6, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice67_p

  subroutine slice77_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_rmi_Call_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_rmi_Call_7d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_rmi_Call__array_slice_m
    call sidl_rmi_Call__array_slice_m(src, 7, numElem, srcStart, srcStride,    &
      newLower, result)
  end subroutine slice77_p

  subroutine getg7_p(array, index, value)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get_m
    call sidl_rmi_Call__array_get_m(array, index, value)
  end subroutine getg7_p

  subroutine setg7_p(array, index, value)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set_m
    call sidl_rmi_Call__array_set_m(array, index, value)
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
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(sidl_rmi_Call_t), intent(out) :: value
    external sidl_rmi_Call__array_get7_m
    call sidl_rmi_Call__array_get7_m(array, &
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
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(sidl_rmi_Call_t), intent(in) :: value
    external sidl_rmi_Call__array_set7_m
    call sidl_rmi_Call__array_set7_m(array, &
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
    type(sidl_rmi_Call_7d), intent(in) :: src
    type(sidl_rmi_Call_7d), intent(out) :: dest
    integer(sidl_int) :: dim
    external Ca_ary_smartCopyuxys5ilkcx0hr_m
    dim = 7
    call Ca_ary_smartCopyuxys5ilkcx0hr_m(src, 7, dest)
  end subroutine smartCopy7_p

  logical function  isColumnOrder7_p(array)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    external ary_isColumnOrderbmi6s8sbzc4u_m
    call ary_isColumnOrderbmi6s8sbzc4u_m(array, isColumnOrder7_p)
  end function isColumnOrder7_p

  logical function  isRowOrder7_p(array)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    external C_ary_isRowOrderhfaiet9x2d5_o_m
    call C_ary_isRowOrderhfaiet9x2d5_o_m(array, isRowOrder7_p)
  end function isRowOrder7_p

  integer (kind=sidl_int) function  dimen7_p(array)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    external sidl_rmi_Call__array_dimen_m
    call sidl_rmi_Call__array_dimen_m(array, dimen7_p)
  end function dimen7_p

  integer (kind=sidl_int) function  stride7_p(array, index)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_stride_m
    call sidl_rmi_Call__array_stride_m(array, index, stride7_p)
  end function stride7_p

  integer (kind=sidl_int) function  lower7_p(array, index)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_lower_m
    call sidl_rmi_Call__array_lower_m(array, index, lower7_p)
  end function lower7_p

  integer (kind=sidl_int) function  upper7_p(array, index)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_upper_m
    call sidl_rmi_Call__array_upper_m(array, index, upper7_p)
  end function upper7_p

  integer (kind=sidl_int) function  length7_p(array, index)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_rmi_Call__array_length_m
    call sidl_rmi_Call__array_length_m(array, index, length7_p)
  end function length7_p

  subroutine  addRef7_p(array)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    external sidl_rmi_Call__array_addRef_m
    call sidl_rmi_Call__array_addRef_m(array)
  end subroutine addRef7_p

  subroutine  deleteRef7_p(array)
    type(sidl_rmi_Call_7d), intent(in)  :: array
    external Ca_ary_deleteRefrpoidx9is5tpw_m
    call Ca_ary_deleteRefrpoidx9is5tpw_m(array)
  end subroutine deleteRef7_p

  logical function is_null7_p(array)
    type(sidl_rmi_Call_7d), intent(in) :: array
    is_null7_p = (array%d_array .eq. 0)
  end function is_null7_p

  logical function not_null7_p(array)
    type(sidl_rmi_Call_7d), intent(in) :: array
    not_null7_p = (array%d_array .ne. 0)
  end function not_null7_p

  subroutine set_null7_p(array)
    type(sidl_rmi_Call_7d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null7_p

  subroutine castsidl_rmi_Call1dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_rmi_Call_1d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_rmi_Call1dToGeneric_p

  subroutine castsidl_rmi_Call2dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_rmi_Call_2d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_rmi_Call2dToGeneric_p

  subroutine castsidl_rmi_Call3dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_rmi_Call_3d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_rmi_Call3dToGeneric_p

  subroutine castsidl_rmi_Call4dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_rmi_Call_4d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_rmi_Call4dToGeneric_p

  subroutine castsidl_rmi_Call5dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_rmi_Call_5d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_rmi_Call5dToGeneric_p

  subroutine castsidl_rmi_Call6dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_rmi_Call_6d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_rmi_Call6dToGeneric_p

  subroutine castsidl_rmi_Call7dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_rmi_Call_7d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_rmi_Call7dToGeneric_p


end module sidl_rmi_Call_array
