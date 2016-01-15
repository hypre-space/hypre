! 
! File:          sidl_Loader.F90
! Symbol:        sidl.Loader-v0.9.15
! Symbol Type:   class
! Babel Version: 1.0.0
! Release:       $Name$
! Revision:      @(#) $Id$
! Description:   Client-side module for sidl.Loader
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
! Symbol "sidl.Loader" (version 0.9.15)
! 
! Class <code>Loader</code> manages dyanamic loading and symbol name
! resolution for the sidl runtime system.  The <code>Loader</code> class
! manages a library search path and keeps a record of all libraries
! loaded through this interface, including the initial "global" symbols
! in the main program.
! 
! Unless explicitly set, the <code>Loader</code> uses the default
! <code>sidl.Finder</code> implemented in <code>sidl.DFinder</code>.
! This class searches the filesystem for <code>.scl</code> files when
! trying to find a class. The initial path is taken from the
! environment variable SIDL_DLL_PATH, which is a semi-colon
! separated sequence of URIs as described in class <code>DLL</code>.
! 


#include "sidl_Finder_fAbbrev.h"
#include "sidl_DLL_fAbbrev.h"
#include "sidl_ClassInfo_fAbbrev.h"
#include "sidl_Resolve_fAbbrev.h"
#include "sidl_Loader_fAbbrev.h"
#include "sidl_BaseInterface_fAbbrev.h"
#include "sidl_RuntimeException_fAbbrev.h"
#include "sidl_BaseException_fAbbrev.h"
#include "sidl_BaseClass_fAbbrev.h"
#include "sidl_Scope_fAbbrev.h"

module sidl_Loader

  use sidl
  use sidl_Finder_type
  use sidl_DLL_type
  use sidl_ClassInfo_type
  use sidl_Loader_type
  use sidl_BaseInterface_type
  use sidl_RuntimeException_type
  use sidl_BaseException_type
  use sidl_BaseClass_type
  use sidl_rmi_Call_type
  use sidl_rmi_Return_type
  use sidl_rmi_Ticket_type

  private :: cast_0, cast_1, cast_2, cast_3
  interface cast
    module procedure cast_0, cast_1, cast_2, cast_3
  end interface

    private :: loadLibrary_s


  interface loadLibrary
    module procedure loadLibrary_s
  end interface
    private :: addDLL_s


  interface addDLL
    module procedure addDLL_s
  end interface
    private :: unloadLibraries_s


  interface unloadLibraries
    module procedure unloadLibraries_s
  end interface
    private :: findLibrary_s


  interface findLibrary
    module procedure findLibrary_s
  end interface
    private :: setSearchPath_s


  interface setSearchPath
    module procedure setSearchPath_s
  end interface
    private :: getSearchPath_s


  interface getSearchPath
    module procedure getSearchPath_s
  end interface
    private :: addSearchPath_s


  interface addSearchPath
    module procedure addSearchPath_s
  end interface
    private :: setFinder_s


  interface setFinder
    module procedure setFinder_s
  end interface
    private :: getFinder_s


  interface getFinder
    module procedure getFinder_s
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
  interface sidl_Loader__set_hooks_static
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



  recursive subroutine loadLibrary_s(uri, loadGlobally, loadLazy, retval,      &
    exception)
    implicit none
    !  in string uri
    character (len=*) , intent(in) :: uri
    !  in bool loadGlobally
    logical , intent(in) :: loadGlobally
    !  in bool loadLazy
    logical , intent(in) :: loadLazy
    !  out sidl.DLL retval
    type(sidl_DLL_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_loadLibrary_m
    call sidl_Loader_loadLibrary_m(uri, loadGlobally, loadLazy, retval,        &
      exception)

  end subroutine loadLibrary_s


  recursive subroutine addDLL_s(dll, exception)
    implicit none
    !  in sidl.DLL dll
    type(sidl_DLL_t) , intent(in) :: dll
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_addDLL_m
    call sidl_Loader_addDLL_m(dll, exception)

  end subroutine addDLL_s


  recursive subroutine unloadLibraries_s(exception)
    implicit none
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_unloadLibraries_m
    call sidl_Loader_unloadLibraries_m(exception)

  end subroutine unloadLibraries_s


  recursive subroutine findLibrary_s(sidl_name, target, lScope, lResolve,      &
    retval, exception)
    implicit none
    !  in string sidl_name
    character (len=*) , intent(in) :: sidl_name
    !  in string target
    character (len=*) , intent(in) :: target
    !  in sidl.Scope lScope
    integer (kind=sidl_enum) , intent(in) :: lScope
    !  in sidl.Resolve lResolve
    integer (kind=sidl_enum) , intent(in) :: lResolve
    !  out sidl.DLL retval
    type(sidl_DLL_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_findLibrary_m
    call sidl_Loader_findLibrary_m(sidl_name, target, lScope, lResolve,        &
      retval, exception)

  end subroutine findLibrary_s


  recursive subroutine setSearchPath_s(path_name, exception)
    implicit none
    !  in string path_name
    character (len=*) , intent(in) :: path_name
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_setSearchPath_m
    call sidl_Loader_setSearchPath_m(path_name, exception)

  end subroutine setSearchPath_s


  recursive subroutine getSearchPath_s(retval, exception)
    implicit none
    !  out string retval
    character (len=*) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_getSearchPath_m
    call sidl_Loader_getSearchPath_m(retval, exception)

  end subroutine getSearchPath_s


  recursive subroutine addSearchPath_s(path_fragment, exception)
    implicit none
    !  in string path_fragment
    character (len=*) , intent(in) :: path_fragment
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_addSearchPath_m
    call sidl_Loader_addSearchPath_m(path_fragment, exception)

  end subroutine addSearchPath_s


  recursive subroutine setFinder_s(f, exception)
    implicit none
    !  in sidl.Finder f
    type(sidl_Finder_t) , intent(in) :: f
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_setFinder_m
    call sidl_Loader_setFinder_m(f, exception)

  end subroutine setFinder_s


  recursive subroutine getFinder_s(retval, exception)
    implicit none
    !  out sidl.Finder retval
    type(sidl_Finder_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_getFinder_m
    call sidl_Loader_getFinder_m(retval, exception)

  end subroutine getFinder_s


  recursive subroutine newLocal_s(retval, exception)
    implicit none
    !  out sidl.Loader retval
    type(sidl_Loader_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_newLocal_m
    call sidl_Loader_newLocal_m(retval, exception)

  end subroutine newLocal_s


  recursive subroutine newRemote_s(self, url, exception)
    implicit none
    !  out sidl.Loader self
    type(sidl_Loader_t) , intent(out) :: self
    !  in string url
    character (len=*) , intent(in) :: url
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_newRemote_m
    call sidl_Loader_newRemote_m(self, url, exception)

  end subroutine newRemote_s


  recursive subroutine rConnect_s(self, url, exception)
    implicit none
    !  out sidl.Loader self
    type(sidl_Loader_t) , intent(out) :: self
    !  in string url
    character (len=*) , intent(in) :: url
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_rConnect_m
    call sidl_Loader_rConnect_m(self, url, exception)

  end subroutine rConnect_s


  recursive subroutine addRef_s(self, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_addRef_m
    call sidl_Loader_addRef_m(self, exception)

  end subroutine addRef_s


  recursive subroutine deleteRef_s(self, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_deleteRef_m
    call sidl_Loader_deleteRef_m(self, exception)

  end subroutine deleteRef_s


  recursive subroutine isSame_s(self, iobj, retval, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  in sidl.BaseInterface iobj
    type(sidl_BaseInterface_t) , intent(in) :: iobj
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_isSame_m
    call sidl_Loader_isSame_m(self, iobj, retval, exception)

  end subroutine isSame_s


  recursive subroutine isType_s(self, name, retval, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  in string name
    character (len=*) , intent(in) :: name
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_isType_m
    call sidl_Loader_isType_m(self, name, retval, exception)

  end subroutine isType_s


  recursive subroutine getClassInfo_s(self, retval, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  out sidl.ClassInfo retval
    type(sidl_ClassInfo_t) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader_getClassInfo_m
    call sidl_Loader_getClassInfo_m(self, retval, exception)

  end subroutine getClassInfo_s

  ! 
  ! Static function to cast from sidl.Loader
  ! to sidl.BaseClass.
  ! 

  subroutine cast_0(oldType, newType, exception)
    implicit none
    type(sidl_Loader_t), intent(in) :: oldType
    type(sidl_BaseClass_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_BaseClass__cast_m

    call sidl_BaseClass__cast_m(oldType, newType, exception)
  end subroutine cast_0

  ! 
  ! Static function to cast from sidl.BaseClass
  ! to sidl.Loader.
  ! 

  subroutine cast_1(oldType, newType, exception)
    implicit none
    type(sidl_BaseClass_t), intent(in) :: oldType
    type(sidl_Loader_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_Loader__cast_m

    call sidl_Loader__cast_m(oldType, newType, exception)
  end subroutine cast_1

  ! 
  ! Static function to cast from sidl.Loader
  ! to sidl.BaseInterface.
  ! 

  subroutine cast_2(oldType, newType, exception)
    implicit none
    type(sidl_Loader_t), intent(in) :: oldType
    type(sidl_BaseInterface_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_BaseInterface__cast_m

    call sidl_BaseInterface__cast_m(oldType, newType, exception)
  end subroutine cast_2

  ! 
  ! Static function to cast from sidl.BaseInterface
  ! to sidl.Loader.
  ! 

  subroutine cast_3(oldType, newType, exception)
    implicit none
    type(sidl_BaseInterface_t), intent(in) :: oldType
    type(sidl_Loader_t), intent(out) :: newType
    type(sidl_BaseInterface_t), intent(out) :: exception
    external sidl_Loader__cast_m

    call sidl_Loader__cast_m(oldType, newType, exception)
  end subroutine cast_3


  recursive subroutine exec_s(self, methodName, inArgs, outArgs, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  in string methodName
    character (len=*) , intent(in) :: methodName
    !  in sidl.rmi.Call inArgs
    type(sidl_rmi_Call_t) , intent(in) :: inArgs
    !  in sidl.rmi.Return outArgs
    type(sidl_rmi_Return_t) , intent(in) :: outArgs
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader__exec_m
    call sidl_Loader__exec_m(self, methodName, inArgs, outArgs, exception)

  end subroutine exec_s

  recursive subroutine getURL_s(self, retval, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  out string retval
    character (len=*) , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader__getURL_m
    call sidl_Loader__getURL_m(self, retval, exception)

  end subroutine getURL_s

  recursive subroutine isRemote_s(self, retval, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader__isRemote_m
    call sidl_Loader__isRemote_m(self, retval, exception)

  end subroutine isRemote_s

  recursive subroutine isLocal_s(self, retval, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  out bool retval
    logical , intent(out) :: retval
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader__isLocal_m
    call sidl_Loader__isLocal_m(self, retval, exception)

  end subroutine isLocal_s

  recursive subroutine set_hooks_s(self, on, exception)
    implicit none
    !  in sidl.Loader self
    type(sidl_Loader_t) , intent(in) :: self
    !  in bool on
    logical , intent(in) :: on
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader__set_hooks_m
    call sidl_Loader__set_hooks_m(self, on, exception)

  end subroutine set_hooks_s

  recursive subroutine set_hooks_static_s(on, exception)
    implicit none
    !  in bool on
    logical , intent(in) :: on
    !  out sidl.BaseInterface exception
    type(sidl_BaseInterface_t) , intent(out) :: exception

    external sidl_Loader__set_hooks_static_m
    call sidl_Loader__set_hooks_static_m(on, exception)

  end subroutine set_hooks_static_s
  logical function is_null_s(ext)
    type(sidl_Loader_t), intent(in) :: ext
    is_null_s = (ext%d_ior .eq. 0)
  end function is_null_s

  logical function not_null_s(ext)
    type(sidl_Loader_t), intent(in) :: ext
    not_null_s = (ext%d_ior .ne. 0)
  end function not_null_s

  subroutine set_null_s(ext)
    type(sidl_Loader_t), intent(out) :: ext
    ext%d_ior = 0
  end subroutine set_null_s


end module sidl_Loader

module sidl_Loader_array
  use sidl
  use sidl_Loader_type
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
    castsidl_Loader1dToGeneric_p, &
    castsidl_Loader2dToGeneric_p, &
    castsidl_Loader3dToGeneric_p, &
    castsidl_Loader4dToGeneric_p, &
    castsidl_Loader5dToGeneric_p, &
    castsidl_Loader6dToGeneric_p, &
    castsidl_Loader7dToGeneric_p
interface cast
  module procedure &
    castsidl_Loader1dToGeneric_p, &
    castsidl_Loader2dToGeneric_p, &
    castsidl_Loader3dToGeneric_p, &
    castsidl_Loader4dToGeneric_p, &
    castsidl_Loader5dToGeneric_p, &
    castsidl_Loader6dToGeneric_p, &
    castsidl_Loader7dToGeneric_p
end interface


contains


  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(sidl_Loader_1d), intent(out) :: array
    external sidl_Loader__array_createCol_m
    call sidl_Loader__array_createCol_m(1, lower, upper, array)
  end subroutine createCol1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow1_p(lower, upper, array)
    integer (kind=sidl_int), dimension(1), intent(in) :: lower
    integer (kind=sidl_int), dimension(1), intent(in) :: upper
    type(sidl_Loader_1d), intent(out) :: array
    external sidl_Loader__array_createRow_m
    call sidl_Loader__array_createRow_m(1, lower, upper, array)
  end subroutine createRow1_p

  subroutine create1d1_p(len, array)
    integer (kind=sidl_int), intent(in) :: len
    type(sidl_Loader_1d), intent(out) :: array
    external sidl_Loader__array_create1d_m
    call sidl_Loader__array_create1d_m(len, array)
  end subroutine create1d1_p

  subroutine copy1_p(src, dest)
    type(sidl_Loader_1d), intent(in) :: src
    type(sidl_Loader_1d), intent(in) :: dest
    external sidl_Loader__array_copy_m
    call sidl_Loader__array_copy_m(src, dest)
  end subroutine copy1_p

  subroutine ensure1_p(src, dim, ordering, result)
    type(sidl_Loader_1d), intent(in)  :: src
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_Loader__array_ensure_m
    call sidl_Loader__array_ensure_m(src, 1, ordering, result)
  end subroutine ensure1_p

  subroutine slice11_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_1d), intent(in)  :: src
    integer (kind=sidl_int), dimension(1), intent(in) :: numElem
    integer (kind=sidl_int), dimension(1), intent(in) :: srcStart, srcStride
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 1, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice11_p

  subroutine getg1_p(array, index, value)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get_m
    call sidl_Loader__array_get_m(array, index, value)
  end subroutine getg1_p

  subroutine setg1_p(array, index, value)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(1) :: index
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set_m
    call sidl_Loader__array_set_m(array, index, value)
  end subroutine setg1_p

  subroutine get1_p(array, &
      i1, &
      value)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get1_m
    call sidl_Loader__array_get1_m(array, &
      i1, &
      value)
  end subroutine get1_p

  subroutine set1_p(array, &
      i1, &
      value)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set1_m
    call sidl_Loader__array_set1_m(array, &
      i1, &
      value)
  end subroutine set1_p

  subroutine smartCopy1_p(src, dest)
    type(sidl_Loader_1d), intent(in) :: src
    type(sidl_Loader_1d), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl_Loader__array_smartCopy_m
    dim = 1
    call sidl_Loader__array_smartCopy_m(src, 1, dest)
  end subroutine smartCopy1_p

  logical function  isColumnOrder1_p(array)
    type(sidl_Loader_1d), intent(in)  :: array
    external ary_isColumnOrderewl3z5ka4ox9_m
    call ary_isColumnOrderewl3z5ka4ox9_m(array, isColumnOrder1_p)
  end function isColumnOrder1_p

  logical function  isRowOrder1_p(array)
    type(sidl_Loader_1d), intent(in)  :: array
    external sidl_Loader__array_isRowOrder_m
    call sidl_Loader__array_isRowOrder_m(array, isRowOrder1_p)
  end function isRowOrder1_p

  integer (kind=sidl_int) function  dimen1_p(array)
    type(sidl_Loader_1d), intent(in)  :: array
    external sidl_Loader__array_dimen_m
    call sidl_Loader__array_dimen_m(array, dimen1_p)
  end function dimen1_p

  integer (kind=sidl_int) function  stride1_p(array, index)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_stride_m
    call sidl_Loader__array_stride_m(array, index, stride1_p)
  end function stride1_p

  integer (kind=sidl_int) function  lower1_p(array, index)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_lower_m
    call sidl_Loader__array_lower_m(array, index, lower1_p)
  end function lower1_p

  integer (kind=sidl_int) function  upper1_p(array, index)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_upper_m
    call sidl_Loader__array_upper_m(array, index, upper1_p)
  end function upper1_p

  integer (kind=sidl_int) function  length1_p(array, index)
    type(sidl_Loader_1d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_length_m
    call sidl_Loader__array_length_m(array, index, length1_p)
  end function length1_p

  subroutine  addRef1_p(array)
    type(sidl_Loader_1d), intent(in)  :: array
    external sidl_Loader__array_addRef_m
    call sidl_Loader__array_addRef_m(array)
  end subroutine addRef1_p

  subroutine  deleteRef1_p(array)
    type(sidl_Loader_1d), intent(in)  :: array
    external sidl_Loader__array_deleteRef_m
    call sidl_Loader__array_deleteRef_m(array)
  end subroutine deleteRef1_p

  logical function is_null1_p(array)
    type(sidl_Loader_1d), intent(in) :: array
    is_null1_p = (array%d_array .eq. 0)
  end function is_null1_p

  logical function not_null1_p(array)
    type(sidl_Loader_1d), intent(in) :: array
    not_null1_p = (array%d_array .ne. 0)
  end function not_null1_p

  subroutine set_null1_p(array)
    type(sidl_Loader_1d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null1_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(sidl_Loader_2d), intent(out) :: array
    external sidl_Loader__array_createCol_m
    call sidl_Loader__array_createCol_m(2, lower, upper, array)
  end subroutine createCol2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow2_p(lower, upper, array)
    integer (kind=sidl_int), dimension(2), intent(in) :: lower
    integer (kind=sidl_int), dimension(2), intent(in) :: upper
    type(sidl_Loader_2d), intent(out) :: array
    external sidl_Loader__array_createRow_m
    call sidl_Loader__array_createRow_m(2, lower, upper, array)
  end subroutine createRow2_p

  subroutine create2dCol2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(sidl_Loader_2d), intent(out) :: array
    external ary_create2dCol_g27_ncg0n_el1_m
    call ary_create2dCol_g27_ncg0n_el1_m(m, n, array)
  end subroutine create2dCol2_p

  subroutine create2dRow2_p(m, n, array)
    integer (kind=sidl_int), intent(in) :: m, n
    type(sidl_Loader_2d), intent(out) :: array
    external ary_create2dRowdyk5_uw8r0wg9f_m
    call ary_create2dRowdyk5_uw8r0wg9f_m(m, n, array)
  end subroutine create2dRow2_p

  subroutine copy2_p(src, dest)
    type(sidl_Loader_2d), intent(in) :: src
    type(sidl_Loader_2d), intent(in) :: dest
    external sidl_Loader__array_copy_m
    call sidl_Loader__array_copy_m(src, dest)
  end subroutine copy2_p

  subroutine ensure2_p(src, dim, ordering, result)
    type(sidl_Loader_2d), intent(in)  :: src
    type(sidl_Loader_2d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_Loader__array_ensure_m
    call sidl_Loader__array_ensure_m(src, 2, ordering, result)
  end subroutine ensure2_p

  subroutine slice12_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 1, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice12_p

  subroutine slice22_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_2d), intent(in)  :: src
    integer (kind=sidl_int), dimension(2), intent(in) :: numElem
    integer (kind=sidl_int), dimension(2), intent(in) :: srcStart, srcStride
    type(sidl_Loader_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 2, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice22_p

  subroutine getg2_p(array, index, value)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get_m
    call sidl_Loader__array_get_m(array, index, value)
  end subroutine getg2_p

  subroutine setg2_p(array, index, value)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(2) :: index
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set_m
    call sidl_Loader__array_set_m(array, index, value)
  end subroutine setg2_p

  subroutine get2_p(array, &
      i1, &
      i2, &
      value)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get2_m
    call sidl_Loader__array_get2_m(array, &
      i1, &
      i2, &
      value)
  end subroutine get2_p

  subroutine set2_p(array, &
      i1, &
      i2, &
      value)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set2_m
    call sidl_Loader__array_set2_m(array, &
      i1, &
      i2, &
      value)
  end subroutine set2_p

  subroutine smartCopy2_p(src, dest)
    type(sidl_Loader_2d), intent(in) :: src
    type(sidl_Loader_2d), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl_Loader__array_smartCopy_m
    dim = 2
    call sidl_Loader__array_smartCopy_m(src, 2, dest)
  end subroutine smartCopy2_p

  logical function  isColumnOrder2_p(array)
    type(sidl_Loader_2d), intent(in)  :: array
    external ary_isColumnOrderewl3z5ka4ox9_m
    call ary_isColumnOrderewl3z5ka4ox9_m(array, isColumnOrder2_p)
  end function isColumnOrder2_p

  logical function  isRowOrder2_p(array)
    type(sidl_Loader_2d), intent(in)  :: array
    external sidl_Loader__array_isRowOrder_m
    call sidl_Loader__array_isRowOrder_m(array, isRowOrder2_p)
  end function isRowOrder2_p

  integer (kind=sidl_int) function  dimen2_p(array)
    type(sidl_Loader_2d), intent(in)  :: array
    external sidl_Loader__array_dimen_m
    call sidl_Loader__array_dimen_m(array, dimen2_p)
  end function dimen2_p

  integer (kind=sidl_int) function  stride2_p(array, index)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_stride_m
    call sidl_Loader__array_stride_m(array, index, stride2_p)
  end function stride2_p

  integer (kind=sidl_int) function  lower2_p(array, index)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_lower_m
    call sidl_Loader__array_lower_m(array, index, lower2_p)
  end function lower2_p

  integer (kind=sidl_int) function  upper2_p(array, index)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_upper_m
    call sidl_Loader__array_upper_m(array, index, upper2_p)
  end function upper2_p

  integer (kind=sidl_int) function  length2_p(array, index)
    type(sidl_Loader_2d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_length_m
    call sidl_Loader__array_length_m(array, index, length2_p)
  end function length2_p

  subroutine  addRef2_p(array)
    type(sidl_Loader_2d), intent(in)  :: array
    external sidl_Loader__array_addRef_m
    call sidl_Loader__array_addRef_m(array)
  end subroutine addRef2_p

  subroutine  deleteRef2_p(array)
    type(sidl_Loader_2d), intent(in)  :: array
    external sidl_Loader__array_deleteRef_m
    call sidl_Loader__array_deleteRef_m(array)
  end subroutine deleteRef2_p

  logical function is_null2_p(array)
    type(sidl_Loader_2d), intent(in) :: array
    is_null2_p = (array%d_array .eq. 0)
  end function is_null2_p

  logical function not_null2_p(array)
    type(sidl_Loader_2d), intent(in) :: array
    not_null2_p = (array%d_array .ne. 0)
  end function not_null2_p

  subroutine set_null2_p(array)
    type(sidl_Loader_2d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null2_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(sidl_Loader_3d), intent(out) :: array
    external sidl_Loader__array_createCol_m
    call sidl_Loader__array_createCol_m(3, lower, upper, array)
  end subroutine createCol3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow3_p(lower, upper, array)
    integer (kind=sidl_int), dimension(3), intent(in) :: lower
    integer (kind=sidl_int), dimension(3), intent(in) :: upper
    type(sidl_Loader_3d), intent(out) :: array
    external sidl_Loader__array_createRow_m
    call sidl_Loader__array_createRow_m(3, lower, upper, array)
  end subroutine createRow3_p

  subroutine copy3_p(src, dest)
    type(sidl_Loader_3d), intent(in) :: src
    type(sidl_Loader_3d), intent(in) :: dest
    external sidl_Loader__array_copy_m
    call sidl_Loader__array_copy_m(src, dest)
  end subroutine copy3_p

  subroutine ensure3_p(src, dim, ordering, result)
    type(sidl_Loader_3d), intent(in)  :: src
    type(sidl_Loader_3d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_Loader__array_ensure_m
    call sidl_Loader__array_ensure_m(src, 3, ordering, result)
  end subroutine ensure3_p

  subroutine slice13_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 1, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice13_p

  subroutine slice23_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(sidl_Loader_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 2, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice23_p

  subroutine slice33_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_3d), intent(in)  :: src
    integer (kind=sidl_int), dimension(3), intent(in) :: numElem
    integer (kind=sidl_int), dimension(3), intent(in) :: srcStart, srcStride
    type(sidl_Loader_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 3, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice33_p

  subroutine getg3_p(array, index, value)
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get_m
    call sidl_Loader__array_get_m(array, index, value)
  end subroutine getg3_p

  subroutine setg3_p(array, index, value)
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(3) :: index
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set_m
    call sidl_Loader__array_set_m(array, index, value)
  end subroutine setg3_p

  subroutine get3_p(array, &
      i1, &
      i2, &
      i3, &
      value)
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get3_m
    call sidl_Loader__array_get3_m(array, &
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
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set3_m
    call sidl_Loader__array_set3_m(array, &
      i1, &
      i2, &
      i3, &
      value)
  end subroutine set3_p

  subroutine smartCopy3_p(src, dest)
    type(sidl_Loader_3d), intent(in) :: src
    type(sidl_Loader_3d), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl_Loader__array_smartCopy_m
    dim = 3
    call sidl_Loader__array_smartCopy_m(src, 3, dest)
  end subroutine smartCopy3_p

  logical function  isColumnOrder3_p(array)
    type(sidl_Loader_3d), intent(in)  :: array
    external ary_isColumnOrderewl3z5ka4ox9_m
    call ary_isColumnOrderewl3z5ka4ox9_m(array, isColumnOrder3_p)
  end function isColumnOrder3_p

  logical function  isRowOrder3_p(array)
    type(sidl_Loader_3d), intent(in)  :: array
    external sidl_Loader__array_isRowOrder_m
    call sidl_Loader__array_isRowOrder_m(array, isRowOrder3_p)
  end function isRowOrder3_p

  integer (kind=sidl_int) function  dimen3_p(array)
    type(sidl_Loader_3d), intent(in)  :: array
    external sidl_Loader__array_dimen_m
    call sidl_Loader__array_dimen_m(array, dimen3_p)
  end function dimen3_p

  integer (kind=sidl_int) function  stride3_p(array, index)
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_stride_m
    call sidl_Loader__array_stride_m(array, index, stride3_p)
  end function stride3_p

  integer (kind=sidl_int) function  lower3_p(array, index)
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_lower_m
    call sidl_Loader__array_lower_m(array, index, lower3_p)
  end function lower3_p

  integer (kind=sidl_int) function  upper3_p(array, index)
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_upper_m
    call sidl_Loader__array_upper_m(array, index, upper3_p)
  end function upper3_p

  integer (kind=sidl_int) function  length3_p(array, index)
    type(sidl_Loader_3d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_length_m
    call sidl_Loader__array_length_m(array, index, length3_p)
  end function length3_p

  subroutine  addRef3_p(array)
    type(sidl_Loader_3d), intent(in)  :: array
    external sidl_Loader__array_addRef_m
    call sidl_Loader__array_addRef_m(array)
  end subroutine addRef3_p

  subroutine  deleteRef3_p(array)
    type(sidl_Loader_3d), intent(in)  :: array
    external sidl_Loader__array_deleteRef_m
    call sidl_Loader__array_deleteRef_m(array)
  end subroutine deleteRef3_p

  logical function is_null3_p(array)
    type(sidl_Loader_3d), intent(in) :: array
    is_null3_p = (array%d_array .eq. 0)
  end function is_null3_p

  logical function not_null3_p(array)
    type(sidl_Loader_3d), intent(in) :: array
    not_null3_p = (array%d_array .ne. 0)
  end function not_null3_p

  subroutine set_null3_p(array)
    type(sidl_Loader_3d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null3_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(sidl_Loader_4d), intent(out) :: array
    external sidl_Loader__array_createCol_m
    call sidl_Loader__array_createCol_m(4, lower, upper, array)
  end subroutine createCol4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow4_p(lower, upper, array)
    integer (kind=sidl_int), dimension(4), intent(in) :: lower
    integer (kind=sidl_int), dimension(4), intent(in) :: upper
    type(sidl_Loader_4d), intent(out) :: array
    external sidl_Loader__array_createRow_m
    call sidl_Loader__array_createRow_m(4, lower, upper, array)
  end subroutine createRow4_p

  subroutine copy4_p(src, dest)
    type(sidl_Loader_4d), intent(in) :: src
    type(sidl_Loader_4d), intent(in) :: dest
    external sidl_Loader__array_copy_m
    call sidl_Loader__array_copy_m(src, dest)
  end subroutine copy4_p

  subroutine ensure4_p(src, dim, ordering, result)
    type(sidl_Loader_4d), intent(in)  :: src
    type(sidl_Loader_4d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_Loader__array_ensure_m
    call sidl_Loader__array_ensure_m(src, 4, ordering, result)
  end subroutine ensure4_p

  subroutine slice14_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 1, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice14_p

  subroutine slice24_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_Loader_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 2, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice24_p

  subroutine slice34_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_Loader_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 3, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice34_p

  subroutine slice44_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_4d), intent(in)  :: src
    integer (kind=sidl_int), dimension(4), intent(in) :: numElem
    integer (kind=sidl_int), dimension(4), intent(in) :: srcStart, srcStride
    type(sidl_Loader_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 4, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice44_p

  subroutine getg4_p(array, index, value)
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get_m
    call sidl_Loader__array_get_m(array, index, value)
  end subroutine getg4_p

  subroutine setg4_p(array, index, value)
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(4) :: index
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set_m
    call sidl_Loader__array_set_m(array, index, value)
  end subroutine setg4_p

  subroutine get4_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get4_m
    call sidl_Loader__array_get4_m(array, &
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
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set4_m
    call sidl_Loader__array_set4_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      value)
  end subroutine set4_p

  subroutine smartCopy4_p(src, dest)
    type(sidl_Loader_4d), intent(in) :: src
    type(sidl_Loader_4d), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl_Loader__array_smartCopy_m
    dim = 4
    call sidl_Loader__array_smartCopy_m(src, 4, dest)
  end subroutine smartCopy4_p

  logical function  isColumnOrder4_p(array)
    type(sidl_Loader_4d), intent(in)  :: array
    external ary_isColumnOrderewl3z5ka4ox9_m
    call ary_isColumnOrderewl3z5ka4ox9_m(array, isColumnOrder4_p)
  end function isColumnOrder4_p

  logical function  isRowOrder4_p(array)
    type(sidl_Loader_4d), intent(in)  :: array
    external sidl_Loader__array_isRowOrder_m
    call sidl_Loader__array_isRowOrder_m(array, isRowOrder4_p)
  end function isRowOrder4_p

  integer (kind=sidl_int) function  dimen4_p(array)
    type(sidl_Loader_4d), intent(in)  :: array
    external sidl_Loader__array_dimen_m
    call sidl_Loader__array_dimen_m(array, dimen4_p)
  end function dimen4_p

  integer (kind=sidl_int) function  stride4_p(array, index)
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_stride_m
    call sidl_Loader__array_stride_m(array, index, stride4_p)
  end function stride4_p

  integer (kind=sidl_int) function  lower4_p(array, index)
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_lower_m
    call sidl_Loader__array_lower_m(array, index, lower4_p)
  end function lower4_p

  integer (kind=sidl_int) function  upper4_p(array, index)
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_upper_m
    call sidl_Loader__array_upper_m(array, index, upper4_p)
  end function upper4_p

  integer (kind=sidl_int) function  length4_p(array, index)
    type(sidl_Loader_4d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_length_m
    call sidl_Loader__array_length_m(array, index, length4_p)
  end function length4_p

  subroutine  addRef4_p(array)
    type(sidl_Loader_4d), intent(in)  :: array
    external sidl_Loader__array_addRef_m
    call sidl_Loader__array_addRef_m(array)
  end subroutine addRef4_p

  subroutine  deleteRef4_p(array)
    type(sidl_Loader_4d), intent(in)  :: array
    external sidl_Loader__array_deleteRef_m
    call sidl_Loader__array_deleteRef_m(array)
  end subroutine deleteRef4_p

  logical function is_null4_p(array)
    type(sidl_Loader_4d), intent(in) :: array
    is_null4_p = (array%d_array .eq. 0)
  end function is_null4_p

  logical function not_null4_p(array)
    type(sidl_Loader_4d), intent(in) :: array
    not_null4_p = (array%d_array .ne. 0)
  end function not_null4_p

  subroutine set_null4_p(array)
    type(sidl_Loader_4d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null4_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(sidl_Loader_5d), intent(out) :: array
    external sidl_Loader__array_createCol_m
    call sidl_Loader__array_createCol_m(5, lower, upper, array)
  end subroutine createCol5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow5_p(lower, upper, array)
    integer (kind=sidl_int), dimension(5), intent(in) :: lower
    integer (kind=sidl_int), dimension(5), intent(in) :: upper
    type(sidl_Loader_5d), intent(out) :: array
    external sidl_Loader__array_createRow_m
    call sidl_Loader__array_createRow_m(5, lower, upper, array)
  end subroutine createRow5_p

  subroutine copy5_p(src, dest)
    type(sidl_Loader_5d), intent(in) :: src
    type(sidl_Loader_5d), intent(in) :: dest
    external sidl_Loader__array_copy_m
    call sidl_Loader__array_copy_m(src, dest)
  end subroutine copy5_p

  subroutine ensure5_p(src, dim, ordering, result)
    type(sidl_Loader_5d), intent(in)  :: src
    type(sidl_Loader_5d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_Loader__array_ensure_m
    call sidl_Loader__array_ensure_m(src, 5, ordering, result)
  end subroutine ensure5_p

  subroutine slice15_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 1, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice15_p

  subroutine slice25_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_Loader_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 2, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice25_p

  subroutine slice35_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_Loader_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 3, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice35_p

  subroutine slice45_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_Loader_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 4, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice45_p

  subroutine slice55_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_5d), intent(in)  :: src
    integer (kind=sidl_int), dimension(5), intent(in) :: numElem
    integer (kind=sidl_int), dimension(5), intent(in) :: srcStart, srcStride
    type(sidl_Loader_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 5, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice55_p

  subroutine getg5_p(array, index, value)
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get_m
    call sidl_Loader__array_get_m(array, index, value)
  end subroutine getg5_p

  subroutine setg5_p(array, index, value)
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(5) :: index
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set_m
    call sidl_Loader__array_set_m(array, index, value)
  end subroutine setg5_p

  subroutine get5_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get5_m
    call sidl_Loader__array_get5_m(array, &
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
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set5_m
    call sidl_Loader__array_set5_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      value)
  end subroutine set5_p

  subroutine smartCopy5_p(src, dest)
    type(sidl_Loader_5d), intent(in) :: src
    type(sidl_Loader_5d), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl_Loader__array_smartCopy_m
    dim = 5
    call sidl_Loader__array_smartCopy_m(src, 5, dest)
  end subroutine smartCopy5_p

  logical function  isColumnOrder5_p(array)
    type(sidl_Loader_5d), intent(in)  :: array
    external ary_isColumnOrderewl3z5ka4ox9_m
    call ary_isColumnOrderewl3z5ka4ox9_m(array, isColumnOrder5_p)
  end function isColumnOrder5_p

  logical function  isRowOrder5_p(array)
    type(sidl_Loader_5d), intent(in)  :: array
    external sidl_Loader__array_isRowOrder_m
    call sidl_Loader__array_isRowOrder_m(array, isRowOrder5_p)
  end function isRowOrder5_p

  integer (kind=sidl_int) function  dimen5_p(array)
    type(sidl_Loader_5d), intent(in)  :: array
    external sidl_Loader__array_dimen_m
    call sidl_Loader__array_dimen_m(array, dimen5_p)
  end function dimen5_p

  integer (kind=sidl_int) function  stride5_p(array, index)
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_stride_m
    call sidl_Loader__array_stride_m(array, index, stride5_p)
  end function stride5_p

  integer (kind=sidl_int) function  lower5_p(array, index)
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_lower_m
    call sidl_Loader__array_lower_m(array, index, lower5_p)
  end function lower5_p

  integer (kind=sidl_int) function  upper5_p(array, index)
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_upper_m
    call sidl_Loader__array_upper_m(array, index, upper5_p)
  end function upper5_p

  integer (kind=sidl_int) function  length5_p(array, index)
    type(sidl_Loader_5d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_length_m
    call sidl_Loader__array_length_m(array, index, length5_p)
  end function length5_p

  subroutine  addRef5_p(array)
    type(sidl_Loader_5d), intent(in)  :: array
    external sidl_Loader__array_addRef_m
    call sidl_Loader__array_addRef_m(array)
  end subroutine addRef5_p

  subroutine  deleteRef5_p(array)
    type(sidl_Loader_5d), intent(in)  :: array
    external sidl_Loader__array_deleteRef_m
    call sidl_Loader__array_deleteRef_m(array)
  end subroutine deleteRef5_p

  logical function is_null5_p(array)
    type(sidl_Loader_5d), intent(in) :: array
    is_null5_p = (array%d_array .eq. 0)
  end function is_null5_p

  logical function not_null5_p(array)
    type(sidl_Loader_5d), intent(in) :: array
    not_null5_p = (array%d_array .ne. 0)
  end function not_null5_p

  subroutine set_null5_p(array)
    type(sidl_Loader_5d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null5_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(sidl_Loader_6d), intent(out) :: array
    external sidl_Loader__array_createCol_m
    call sidl_Loader__array_createCol_m(6, lower, upper, array)
  end subroutine createCol6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow6_p(lower, upper, array)
    integer (kind=sidl_int), dimension(6), intent(in) :: lower
    integer (kind=sidl_int), dimension(6), intent(in) :: upper
    type(sidl_Loader_6d), intent(out) :: array
    external sidl_Loader__array_createRow_m
    call sidl_Loader__array_createRow_m(6, lower, upper, array)
  end subroutine createRow6_p

  subroutine copy6_p(src, dest)
    type(sidl_Loader_6d), intent(in) :: src
    type(sidl_Loader_6d), intent(in) :: dest
    external sidl_Loader__array_copy_m
    call sidl_Loader__array_copy_m(src, dest)
  end subroutine copy6_p

  subroutine ensure6_p(src, dim, ordering, result)
    type(sidl_Loader_6d), intent(in)  :: src
    type(sidl_Loader_6d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_Loader__array_ensure_m
    call sidl_Loader__array_ensure_m(src, 6, ordering, result)
  end subroutine ensure6_p

  subroutine slice16_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 1, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice16_p

  subroutine slice26_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_Loader_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 2, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice26_p

  subroutine slice36_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_Loader_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 3, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice36_p

  subroutine slice46_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_Loader_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 4, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice46_p

  subroutine slice56_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_Loader_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 5, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice56_p

  subroutine slice66_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_6d), intent(in)  :: src
    integer (kind=sidl_int), dimension(6), intent(in) :: numElem
    integer (kind=sidl_int), dimension(6), intent(in) :: srcStart, srcStride
    type(sidl_Loader_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 6, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice66_p

  subroutine getg6_p(array, index, value)
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get_m
    call sidl_Loader__array_get_m(array, index, value)
  end subroutine getg6_p

  subroutine setg6_p(array, index, value)
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(6) :: index
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set_m
    call sidl_Loader__array_set_m(array, index, value)
  end subroutine setg6_p

  subroutine get6_p(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get6_m
    call sidl_Loader__array_get6_m(array, &
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
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set6_m
    call sidl_Loader__array_set6_m(array, &
      i1, &
      i2, &
      i3, &
      i4, &
      i5, &
      i6, &
      value)
  end subroutine set6_p

  subroutine smartCopy6_p(src, dest)
    type(sidl_Loader_6d), intent(in) :: src
    type(sidl_Loader_6d), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl_Loader__array_smartCopy_m
    dim = 6
    call sidl_Loader__array_smartCopy_m(src, 6, dest)
  end subroutine smartCopy6_p

  logical function  isColumnOrder6_p(array)
    type(sidl_Loader_6d), intent(in)  :: array
    external ary_isColumnOrderewl3z5ka4ox9_m
    call ary_isColumnOrderewl3z5ka4ox9_m(array, isColumnOrder6_p)
  end function isColumnOrder6_p

  logical function  isRowOrder6_p(array)
    type(sidl_Loader_6d), intent(in)  :: array
    external sidl_Loader__array_isRowOrder_m
    call sidl_Loader__array_isRowOrder_m(array, isRowOrder6_p)
  end function isRowOrder6_p

  integer (kind=sidl_int) function  dimen6_p(array)
    type(sidl_Loader_6d), intent(in)  :: array
    external sidl_Loader__array_dimen_m
    call sidl_Loader__array_dimen_m(array, dimen6_p)
  end function dimen6_p

  integer (kind=sidl_int) function  stride6_p(array, index)
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_stride_m
    call sidl_Loader__array_stride_m(array, index, stride6_p)
  end function stride6_p

  integer (kind=sidl_int) function  lower6_p(array, index)
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_lower_m
    call sidl_Loader__array_lower_m(array, index, lower6_p)
  end function lower6_p

  integer (kind=sidl_int) function  upper6_p(array, index)
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_upper_m
    call sidl_Loader__array_upper_m(array, index, upper6_p)
  end function upper6_p

  integer (kind=sidl_int) function  length6_p(array, index)
    type(sidl_Loader_6d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_length_m
    call sidl_Loader__array_length_m(array, index, length6_p)
  end function length6_p

  subroutine  addRef6_p(array)
    type(sidl_Loader_6d), intent(in)  :: array
    external sidl_Loader__array_addRef_m
    call sidl_Loader__array_addRef_m(array)
  end subroutine addRef6_p

  subroutine  deleteRef6_p(array)
    type(sidl_Loader_6d), intent(in)  :: array
    external sidl_Loader__array_deleteRef_m
    call sidl_Loader__array_deleteRef_m(array)
  end subroutine deleteRef6_p

  logical function is_null6_p(array)
    type(sidl_Loader_6d), intent(in) :: array
    is_null6_p = (array%d_array .eq. 0)
  end function is_null6_p

  logical function not_null6_p(array)
    type(sidl_Loader_6d), intent(in) :: array
    not_null6_p = (array%d_array .ne. 0)
  end function not_null6_p

  subroutine set_null6_p(array)
    type(sidl_Loader_6d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null6_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createCol7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(sidl_Loader_7d), intent(out) :: array
    external sidl_Loader__array_createCol_m
    call sidl_Loader__array_createCol_m(7, lower, upper, array)
  end subroutine createCol7_p

  ! 
  ! The size of lower determines the dimension of the
  ! array.
  ! 

  subroutine createRow7_p(lower, upper, array)
    integer (kind=sidl_int), dimension(7), intent(in) :: lower
    integer (kind=sidl_int), dimension(7), intent(in) :: upper
    type(sidl_Loader_7d), intent(out) :: array
    external sidl_Loader__array_createRow_m
    call sidl_Loader__array_createRow_m(7, lower, upper, array)
  end subroutine createRow7_p

  subroutine copy7_p(src, dest)
    type(sidl_Loader_7d), intent(in) :: src
    type(sidl_Loader_7d), intent(in) :: dest
    external sidl_Loader__array_copy_m
    call sidl_Loader__array_copy_m(src, dest)
  end subroutine copy7_p

  subroutine ensure7_p(src, dim, ordering, result)
    type(sidl_Loader_7d), intent(in)  :: src
    type(sidl_Loader_7d), intent(out) :: result
    integer (kind=sidl_int), intent(in) :: dim, ordering
    external sidl_Loader__array_ensure_m
    call sidl_Loader__array_ensure_m(src, 7, ordering, result)
  end subroutine ensure7_p

  subroutine slice17_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_Loader_1d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 1, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice17_p

  subroutine slice27_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_Loader_2d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 2, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice27_p

  subroutine slice37_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_Loader_3d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 3, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice37_p

  subroutine slice47_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_Loader_4d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 4, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice47_p

  subroutine slice57_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_Loader_5d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 5, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice57_p

  subroutine slice67_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_Loader_6d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 6, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice67_p

  subroutine slice77_p(src, numElem, srcStart, srcStride, newLower, result)
    type(sidl_Loader_7d), intent(in)  :: src
    integer (kind=sidl_int), dimension(7), intent(in) :: numElem
    integer (kind=sidl_int), dimension(7), intent(in) :: srcStart, srcStride
    type(sidl_Loader_7d), intent(out) :: result
    integer (kind=sidl_int), dimension(:), intent(in) :: newLower
    external sidl_Loader__array_slice_m
    call sidl_Loader__array_slice_m(src, 7, numElem, srcStart, srcStride,      &
      newLower, result)
  end subroutine slice77_p

  subroutine getg7_p(array, index, value)
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get_m
    call sidl_Loader__array_get_m(array, index, value)
  end subroutine getg7_p

  subroutine setg7_p(array, index, value)
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in), dimension(7) :: index
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set_m
    call sidl_Loader__array_set_m(array, index, value)
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
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(sidl_Loader_t), intent(out) :: value
    external sidl_Loader__array_get7_m
    call sidl_Loader__array_get7_m(array, &
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
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: i1
    integer (kind=sidl_int), intent(in) :: i2
    integer (kind=sidl_int), intent(in) :: i3
    integer (kind=sidl_int), intent(in) :: i4
    integer (kind=sidl_int), intent(in) :: i5
    integer (kind=sidl_int), intent(in) :: i6
    integer (kind=sidl_int), intent(in) :: i7
    type(sidl_Loader_t), intent(in) :: value
    external sidl_Loader__array_set7_m
    call sidl_Loader__array_set7_m(array, &
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
    type(sidl_Loader_7d), intent(in) :: src
    type(sidl_Loader_7d), intent(out) :: dest
    integer(sidl_int) :: dim
    external sidl_Loader__array_smartCopy_m
    dim = 7
    call sidl_Loader__array_smartCopy_m(src, 7, dest)
  end subroutine smartCopy7_p

  logical function  isColumnOrder7_p(array)
    type(sidl_Loader_7d), intent(in)  :: array
    external ary_isColumnOrderewl3z5ka4ox9_m
    call ary_isColumnOrderewl3z5ka4ox9_m(array, isColumnOrder7_p)
  end function isColumnOrder7_p

  logical function  isRowOrder7_p(array)
    type(sidl_Loader_7d), intent(in)  :: array
    external sidl_Loader__array_isRowOrder_m
    call sidl_Loader__array_isRowOrder_m(array, isRowOrder7_p)
  end function isRowOrder7_p

  integer (kind=sidl_int) function  dimen7_p(array)
    type(sidl_Loader_7d), intent(in)  :: array
    external sidl_Loader__array_dimen_m
    call sidl_Loader__array_dimen_m(array, dimen7_p)
  end function dimen7_p

  integer (kind=sidl_int) function  stride7_p(array, index)
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_stride_m
    call sidl_Loader__array_stride_m(array, index, stride7_p)
  end function stride7_p

  integer (kind=sidl_int) function  lower7_p(array, index)
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_lower_m
    call sidl_Loader__array_lower_m(array, index, lower7_p)
  end function lower7_p

  integer (kind=sidl_int) function  upper7_p(array, index)
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_upper_m
    call sidl_Loader__array_upper_m(array, index, upper7_p)
  end function upper7_p

  integer (kind=sidl_int) function  length7_p(array, index)
    type(sidl_Loader_7d), intent(in)  :: array
    integer (kind=sidl_int), intent(in) :: index
    external sidl_Loader__array_length_m
    call sidl_Loader__array_length_m(array, index, length7_p)
  end function length7_p

  subroutine  addRef7_p(array)
    type(sidl_Loader_7d), intent(in)  :: array
    external sidl_Loader__array_addRef_m
    call sidl_Loader__array_addRef_m(array)
  end subroutine addRef7_p

  subroutine  deleteRef7_p(array)
    type(sidl_Loader_7d), intent(in)  :: array
    external sidl_Loader__array_deleteRef_m
    call sidl_Loader__array_deleteRef_m(array)
  end subroutine deleteRef7_p

  logical function is_null7_p(array)
    type(sidl_Loader_7d), intent(in) :: array
    is_null7_p = (array%d_array .eq. 0)
  end function is_null7_p

  logical function not_null7_p(array)
    type(sidl_Loader_7d), intent(in) :: array
    not_null7_p = (array%d_array .ne. 0)
  end function not_null7_p

  subroutine set_null7_p(array)
    type(sidl_Loader_7d), intent(out) :: array
    array%d_array = 0
  end subroutine set_null7_p

  subroutine castsidl_Loader1dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_Loader_1d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_Loader1dToGeneric_p

  subroutine castsidl_Loader2dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_Loader_2d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_Loader2dToGeneric_p

  subroutine castsidl_Loader3dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_Loader_3d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_Loader3dToGeneric_p

  subroutine castsidl_Loader4dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_Loader_4d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_Loader4dToGeneric_p

  subroutine castsidl_Loader5dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_Loader_5d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_Loader5dToGeneric_p

  subroutine castsidl_Loader6dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_Loader_6d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_Loader6dToGeneric_p

  subroutine castsidl_Loader7dToGeneric_p(oldType, newType)
    type(sidl__array), intent(out) :: newType
    type(sidl_Loader_7d), intent(in) :: oldType
    newType%d_array = oldType%d_array
  end subroutine castsidl_Loader7dToGeneric_p


end module sidl_Loader_array
