// 
// File:          sidl_rmi_InstanceHandle.cc
// Symbol:        sidl.rmi.InstanceHandle-v0.9.3
// Symbol Type:   interface
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.rmi.InstanceHandle
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
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.rmi.InstanceHandle-v0.9.3.xml
// 

#ifndef included_sidl_rmi_InstanceHandle_hh
#include "sidl_rmi_InstanceHandle.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif
#ifndef included_sidl_BaseException_hh
#include "sidl_BaseException.hh"
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
#ifndef included_sidl_rmi_Invocation_hh
#include "sidl_rmi_Invocation.hh"
#endif
#ifndef included_sidl_rmi_NetworkException_hh
#include "sidl_rmi_NetworkException.hh"
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
ucxx::sidl::rmi::InstanceHandle::isSame( /* in */::ucxx::sidl::BaseInterface 
  iobj )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  sidl_bool _local_result;
  struct sidl_BaseInterface__object* _local_iobj = reinterpret_cast< struct 
    sidl_BaseInterface__object* > ( iobj._get_ior() ? ((*((reinterpret_cast< 
    struct sidl_BaseInterface__object * > 
    (iobj._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (iobj._get_ior()))->d_object,
    "sidl.BaseInterface")) : 0);
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isSame))(loc_self->d_object,
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
ucxx::sidl::rmi::InstanceHandle::queryInt( /* in */const ::std::string& name )
throw ()

{
  ::ucxx::sidl::BaseInterface _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::BaseInterface( 
    (*(loc_self->d_epv->f_queryInt))(loc_self->d_object,
    /* in */ name.c_str() ), false);
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
ucxx::sidl::rmi::InstanceHandle::isType( /* in */const ::std::string& name )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  sidl_bool _local_result;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isType))(loc_self->d_object,
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
ucxx::sidl::rmi::InstanceHandle::getClassInfo(  )
throw ()

{
  ::ucxx::sidl::ClassInfo _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::ClassInfo( 
    (*(loc_self->d_epv->f_getClassInfo))(loc_self->d_object ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * initialize a connection (intended for use by the ProtocolFactory) 
 */
bool
ucxx::sidl::rmi::InstanceHandle::init( /* in */const ::std::string& url,
  /* in */const ::std::string& typeName )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  sidl_bool _local_result;
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_initCreate))(loc_self->d_object,
    /* in */ url.c_str(), /* in */ typeName.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.rmi.NetworkException")) != 0 ) {
      struct sidl_rmi_NetworkException__object * _realtype = reinterpret_cast< 
        struct sidl_rmi_NetworkException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::rmi::NetworkException( _realtype, false );
    }
  }
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * initialize a connection (intended for use by the ProtocolFactory) 
 */
bool
ucxx::sidl::rmi::InstanceHandle::init( /* in */const ::std::string& url )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  sidl_bool _local_result;
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_initConnect))(loc_self->d_object,
    /* in */ url.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.rmi.NetworkException")) != 0 ) {
      struct sidl_rmi_NetworkException__object * _realtype = reinterpret_cast< 
        struct sidl_rmi_NetworkException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::rmi::NetworkException( _realtype, false );
    }
  }
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * return the name of the protocol 
 */
::std::string
ucxx::sidl::rmi::InstanceHandle::getProtocol(  )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  ::std::string _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  char * _local_result;
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_getProtocol))(loc_self->d_object,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.rmi.NetworkException")) != 0 ) {
      struct sidl_rmi_NetworkException__object * _realtype = reinterpret_cast< 
        struct sidl_rmi_NetworkException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::rmi::NetworkException( _realtype, false );
    }
  }
  if (_local_result) {
    _result = _local_result;
    free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * return the session ID 
 */
::std::string
ucxx::sidl::rmi::InstanceHandle::getObjectID(  )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  ::std::string _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  char * _local_result;
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_getObjectID))(loc_self->d_object,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.rmi.NetworkException")) != 0 ) {
      struct sidl_rmi_NetworkException__object * _realtype = reinterpret_cast< 
        struct sidl_rmi_NetworkException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::rmi::NetworkException( _realtype, false );
    }
  }
  if (_local_result) {
    _result = _local_result;
    free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * return the full URL for this object, takes the form: 
 *  protocol://server:port/class/objectID
 */
::std::string
ucxx::sidl::rmi::InstanceHandle::getURL(  )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  ::std::string _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  char * _local_result;
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_getURL))(loc_self->d_object,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.rmi.NetworkException")) != 0 ) {
      struct sidl_rmi_NetworkException__object * _realtype = reinterpret_cast< 
        struct sidl_rmi_NetworkException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::rmi::NetworkException( _realtype, false );
    }
  }
  if (_local_result) {
    _result = _local_result;
    free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * create a handle to invoke a named method 
 */
::ucxx::sidl::rmi::Invocation
ucxx::sidl::rmi::InstanceHandle::createInvocation( /* in */const ::std::string& 
  methodName )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  ::ucxx::sidl::rmi::Invocation _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::rmi::Invocation( 
    (*(loc_self->d_epv->f_createInvocation))(loc_self->d_object,
    /* in */ methodName.c_str(), &_exception ), false);
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.rmi.NetworkException")) != 0 ) {
      struct sidl_rmi_NetworkException__object * _realtype = reinterpret_cast< 
        struct sidl_rmi_NetworkException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::rmi::NetworkException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * closes the connection (called be destructor, if not done explicitly) 
 * returns true if successful, false otherwise (including subsequent calls)
 */
bool
ucxx::sidl::rmi::InstanceHandle::close(  )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "sidl.rmi.InstanceHandle");
  sidl_bool _local_result;
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_close))(loc_self->d_object,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.rmi.NetworkException")) != 0 ) {
      struct sidl_rmi_NetworkException__object * _realtype = reinterpret_cast< 
        struct sidl_rmi_NetworkException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::rmi::NetworkException( _realtype, false );
    }
  }
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// copy constructor
ucxx::sidl::rmi::InstanceHandle::InstanceHandle ( const 
  ::ucxx::sidl::rmi::InstanceHandle& original ) {
  d_self = original._cast("sidl.rmi.InstanceHandle");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::sidl::rmi::InstanceHandle&
ucxx::sidl::rmi::InstanceHandle::operator=( const 
  ::ucxx::sidl::rmi::InstanceHandle& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("sidl.rmi.InstanceHandle");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::sidl::rmi::InstanceHandle::InstanceHandle ( 
  ::ucxx::sidl::rmi::InstanceHandle::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::sidl::rmi::InstanceHandle::InstanceHandle ( 
  ::ucxx::sidl::rmi::InstanceHandle::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::sidl::rmi::InstanceHandle::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< 
      void*>((*loc_self->d_epv->f__cast)(loc_self->d_object, type));
  }
  return ptr;
}

