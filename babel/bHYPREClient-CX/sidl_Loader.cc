// 
// File:          sidl_Loader.cc
// Symbol:        sidl.Loader-v0.9.3
// Symbol Type:   class
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.Loader
// 
// Copyright (c) 2000-2002, The Regents of the University of California.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the Components Team <components@llnl.gov>
// All rights reserved.
// 
// This file is part of Babel. For more information, see
// http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
// for Our Notice and the LICENSE file for the GNU Lesser General Public
// License.
// 
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License (as published by
// the Free Software Foundation) version 2.1 dated February 1999.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// conditions of the GNU Lesser General Public License for more details.
// 
// You should have recieved a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.Loader-v0.9.3.xml
// 

#ifndef included_sidl_Loader_hh
#include "sidl_Loader.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif
#include "sidl_String.h"
// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif
#ifndef included_sidl_DLL_hh
#include "sidl_DLL.hh"
#endif
#ifndef included_sidl_Finder_hh
#include "sidl_Finder.hh"
#endif
#ifndef included_sidl_Resolve_hh
#include "sidl_Resolve.hh"
#endif
#ifndef included_sidl_Scope_hh
#include "sidl_Scope.hh"
#endif


//////////////////////////////////////////////////
// 
// User Defined Methods
// 


/**
 * Load the specified library if it has not already been loaded.
 * The URI format is defined in class <code>DLL</code>.  The search
 * path is not searched to resolve the library name.
 * 
 * @param uri          the URI to load. This can be a .la file
 *                     (a metadata file produced by libtool) or
 *                     a shared library binary (i.e., .so,
 *                     .dll or whatever is appropriate for your
 *                     OS)
 * @param loadGlobally <code>true</code> means that the shared
 *                     library symbols will be loaded into the
 *                     global namespace; <code>false</code> 
 *                     means they will be loaded into a 
 *                     private namespace. Some operating systems
 *                     may not be able to honor the value presented
 *                     here.
 * @param loadLazy     <code>true</code> instructs the loader to
 *                     that symbols can be resolved as needed (lazy)
 *                     instead of requiring everything to be resolved
 *                     now.
 * @return if the load was successful, a non-NULL DLL object is returned.
 */
::ucxx::sidl::DLL
ucxx::sidl::Loader::loadLibrary( /* in */const ::std::string& uri,
  /* in */bool loadGlobally, /* in */bool loadLazy )
throw ()

{
  ::ucxx::sidl::DLL _result;
  sidl_bool _local_loadGlobally = loadGlobally;
  sidl_bool _local_loadLazy = loadLazy;
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::DLL( ( _get_sepv()->f_loadLibrary)( /* in */ 
    uri.c_str(), /* in */ _local_loadGlobally, /* in */ _local_loadLazy ),
    false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Append the specified DLL to the beginning of the list of already
 * loaded DLLs.
 */
void
ucxx::sidl::Loader::addDLL( /* in */::ucxx::sidl::DLL dll )
throw ()

{

  struct sidl_DLL__object* _local_dll = dll._get_ior();
  /*pack args to dispatch to ior*/
  ( _get_sepv()->f_addDLL)( /* in */ _local_dll );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
}



/**
 * Find a DLL containing the specified information for a sidl
 * class. This method searches SCL files in the search path looking
 * for a shared library that contains the client-side or IOR
 * for a particular sidl class.
 * 
 * This call is implemented by calling the current
 * <code>Finder</code>. The default finder searches the local
 * file system for <code>.scl</code> files to locate the
 * target class/interface.
 * 
 * @param sidl_name  the fully qualified (long) name of the
 *                   class/interface to be found. Package names
 *                   are separated by period characters from each
 *                   other and the class/interface name.
 * @param target     to find a client-side binding, this is
 *                   normally the name of the language.
 *                   To find the implementation of a class
 *                   in order to make one, you should pass
 *                   the string "ior/impl" here.
 * @param lScope     this specifies whether the symbols should
 *                   be loaded into the global scope, a local
 *                   scope, or use the setting in the SCL file.
 * @param lResolve   this specifies whether symbols should be
 *                   resolved as needed (LAZY), completely
 *                   resolved at load time (NOW), or use the
 *                   setting from the SCL file.
 * @return a non-NULL object means the search was successful.
 *         The DLL has already been added.
 */
::ucxx::sidl::DLL
ucxx::sidl::Loader::findLibrary( /* in */const ::std::string& sidl_name,
  /* in */const ::std::string& target, /* in */::ucxx::sidl::Scope lScope,
  /* in */::ucxx::sidl::Resolve lResolve )
throw ()

{
  ::ucxx::sidl::DLL _result;
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::DLL( ( _get_sepv()->f_findLibrary)( /* in */ 
    sidl_name.c_str(), /* in */ target.c_str(),
    /* in */ (enum sidl_Scope__enum)lScope,
    /* in */ (enum sidl_Resolve__enum)lResolve ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the search path, which is a semi-colon separated sequence of
 * URIs as described in class <code>DLL</code>.  This method will
 * invalidate any existing search path.
 * 
 * This updates the search path in the current <code>Finder</code>.
 */
void
ucxx::sidl::Loader::setSearchPath( /* in */const ::std::string& path_name )
throw ()

{

  /*pack args to dispatch to ior*/
  ( _get_sepv()->f_setSearchPath)( /* in */ path_name.c_str() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
}



/**
 * Return the current search path.  The default
 * <code>Finder</code> initializes the search path
 * from environment variable SIDL_DLL_PATH.
 * 
 */
::std::string
ucxx::sidl::Loader::getSearchPath(  )
throw ()

{
  ::std::string _result;
  char * _local_result;
  /*pack args to dispatch to ior*/
  _local_result = ( _get_sepv()->f_getSearchPath)(  );
  /*dispatch to ior*/
  if (_local_result) {
    _result = _local_result;
    free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Append the specified path fragment to the beginning of the
 * current search path.  This method operates on the Loader's
 * current <code>Finder</code>. This will add a path to the
 * current search path. Normally, the search path is initialized
 * from the SIDL_DLL_PATH environment variable.
 */
void
ucxx::sidl::Loader::addSearchPath( /* in */const ::std::string& path_fragment )
throw ()

{

  /*pack args to dispatch to ior*/
  ( _get_sepv()->f_addSearchPath)( /* in */ path_fragment.c_str() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
}



/**
 * This method sets the <code>Finder</code> that
 * <code>Loader</code> will use to find DLLs.  If no
 * <code>Finder</code> is set or if NULL is passed in, the Default
 * Finder <code>DFinder</code> will be used.
 * 
 * Future calls to <code>findLibrary</code>,
 * <code>addSearchPath</code>, <code>getSearchPath</code>, and
 * <code>setSearchPath</code> are deligated to the
 * <code>Finder</code> set here.
 */
void
ucxx::sidl::Loader::setFinder( /* in */::ucxx::sidl::Finder f )
throw ()

{

  struct sidl_Finder__object* _local_f = reinterpret_cast< struct 
    sidl_Finder__object* > ( f._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (f._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (f._get_ior()))->d_object,
    "sidl.Finder")) : 0);
  /*pack args to dispatch to ior*/
  ( _get_sepv()->f_setFinder)( /* in */ _local_f );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
}



/**
 * This method gets the <code>Finder</code> that <code>Loader</code>
 * uses to find DLLs.  
 */
::ucxx::sidl::Finder
ucxx::sidl::Loader::getFinder(  )
throw ()

{
  ::ucxx::sidl::Finder _result;
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::Finder( ( _get_sepv()->f_getFinder)(  ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::sidl::Loader::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
throw ()

{
  bool _result;
  ior_t* loc_self = _get_ior();
  sidl_bool _local_result;
  struct sidl_BaseInterface__object* _local_iobj = reinterpret_cast< struct 
    sidl_BaseInterface__object* > ( iobj._get_ior() ? ((*((reinterpret_cast< 
    struct sidl_BaseInterface__object * > 
    (iobj._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (iobj._get_ior()))->d_object,
    "sidl.BaseInterface")) : 0);
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isSame))(loc_self,
    /* in */ _local_iobj );
  /*dispatch to ior*/
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
::ucxx::sidl::BaseInterface
ucxx::sidl::Loader::queryInt( /* in */const ::std::string& name )
throw ()

{
  ::ucxx::sidl::BaseInterface _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::BaseInterface( 
    (*(loc_self->d_epv->f_queryInt))(loc_self, /* in */ name.c_str() ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
bool
ucxx::sidl::Loader::isType( /* in */const ::std::string& name )
throw ()

{
  bool _result;
  ior_t* loc_self = _get_ior();
  sidl_bool _local_result;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isType))(loc_self,
    /* in */ name.c_str() );
  /*dispatch to ior*/
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return the meta-data about the class implementing this interface.
 */
::ucxx::sidl::ClassInfo
ucxx::sidl::Loader::getClassInfo(  )
throw ()

{
  ::ucxx::sidl::ClassInfo _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::ClassInfo( 
    (*(loc_self->d_epv->f_getClassInfo))(loc_self ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::ucxx::sidl::Loader
ucxx::sidl::Loader::_create() {
  ::ucxx::sidl::Loader self( (*_get_ext()->createObject)(), false );
  return self;
}

// copy constructor
ucxx::sidl::Loader::Loader ( const ::ucxx::sidl::Loader& original ) {
  d_self = original._cast("sidl.Loader");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::sidl::Loader&
ucxx::sidl::Loader::operator=( const ::ucxx::sidl::Loader& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("sidl.Loader");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::sidl::Loader::Loader ( ::ucxx::sidl::Loader::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::sidl::Loader::Loader ( ::ucxx::sidl::Loader::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::sidl::Loader::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::sidl::Loader::ext_t * ucxx::sidl::Loader::s_ext = 0;

// private static method to get static data type
const ::ucxx::sidl::Loader::ext_t *
ucxx::sidl::Loader::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
    s_ext = sidl_Loader__externals();
  }
  return s_ext;
}

