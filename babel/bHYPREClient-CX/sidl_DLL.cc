// 
// File:          sidl_DLL.cc
// Symbol:        sidl.DLL-v0.9.3
// Symbol Type:   class
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.DLL
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
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.DLL-v0.9.3.xml
// 

#ifndef included_sidl_DLL_hh
#include "sidl_DLL.hh"
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
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_ClassInfo_hh
#include "sidl_ClassInfo.hh"
#endif


//////////////////////////////////////////////////
// 
// User Defined Methods
// 


/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::sidl::DLL::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
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
ucxx::sidl::DLL::queryInt( /* in */const ::std::string& name )
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
ucxx::sidl::DLL::isType( /* in */const ::std::string& name )
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
ucxx::sidl::DLL::getClassInfo(  )
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



/**
 * Load a dynamic link library using the specified URI.  The
 * URI may be of the form "main:", "lib:", "file:", "ftp:", or
 * "http:".  A URI that starts with any other protocol string
 * is assumed to be a file name.  The "main:" URI creates a
 * library that allows access to global symbols in the running
 * program's main address space.  The "lib:X" URI converts the
 * library "X" into a platform-specific name (e.g., libX.so) and
 * loads that library.  The "file:" URI opens the DLL from the
 * specified file path.  The "ftp:" and "http:" URIs copy the
 * specified library from the remote site into a local temporary
 * file and open that file.  This method returns true if the
 * DLL was loaded successfully and false otherwise.  Note that
 * the "ftp:" and "http:" protocols are valid only if the W3C
 * WWW library is available.
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
 *                     now (at load time).
 */
bool
ucxx::sidl::DLL::loadLibrary( /* in */const ::std::string& uri,
  /* in */bool loadGlobally, /* in */bool loadLazy )
throw ()

{
  bool _result;
  ior_t* loc_self = _get_ior();
  sidl_bool _local_result;
  sidl_bool _local_loadGlobally = loadGlobally;
  sidl_bool _local_loadLazy = loadLazy;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_loadLibrary))(loc_self,
    /* in */ uri.c_str(), /* in */ _local_loadGlobally,
    /* in */ _local_loadLazy );
  /*dispatch to ior*/
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Get the library name.  This is the name used to load the
 * library in <code>loadLibrary</code> except that all file names
 * contain the "file:" protocol.
 */
::std::string
ucxx::sidl::DLL::getName(  )
throw ()

{
  ::std::string _result;
  ior_t* loc_self = _get_ior();
  char * _local_result;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_getName))(loc_self );
  /*dispatch to ior*/
  if (_local_result) {
    _result = _local_result;
    free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Lookup a symbol from the DLL and return the associated pointer.
 * A null value is returned if the name does not exist.
 */
void*
ucxx::sidl::DLL::lookupSymbol( /* in */const ::std::string& linker_name )
throw ()

{
  void* _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_lookupSymbol))(loc_self,
    /* in */ linker_name.c_str() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Create an instance of the sidl class.  If the class constructor
 * is not defined in this DLL, then return null.
 */
::ucxx::sidl::BaseClass
ucxx::sidl::DLL::createClass( /* in */const ::std::string& sidl_name )
throw ()

{
  ::ucxx::sidl::BaseClass _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::BaseClass( 
    (*(loc_self->d_epv->f_createClass))(loc_self, /* in */ sidl_name.c_str() ),
    false);
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
::ucxx::sidl::DLL
ucxx::sidl::DLL::_create() {
  ::ucxx::sidl::DLL self( (*_get_ext()->createObject)(), false );
  return self;
}

// copy constructor
ucxx::sidl::DLL::DLL ( const ::ucxx::sidl::DLL& original ) {
  d_self = original._cast("sidl.DLL");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::sidl::DLL&
ucxx::sidl::DLL::operator=( const ::ucxx::sidl::DLL& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("sidl.DLL");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::sidl::DLL::DLL ( ::ucxx::sidl::DLL::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::sidl::DLL::DLL ( ::ucxx::sidl::DLL::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::sidl::DLL::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::sidl::DLL::ext_t * ucxx::sidl::DLL::s_ext = 0;

// private static method to get static data type
const ::ucxx::sidl::DLL::ext_t *
ucxx::sidl::DLL::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
    s_ext = sidl_DLL__externals();
  }
  return s_ext;
}

