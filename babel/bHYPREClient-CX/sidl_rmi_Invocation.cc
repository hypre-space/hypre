// 
// File:          sidl_rmi_Invocation.cc
// Symbol:        sidl.rmi.Invocation-v0.9.3
// Symbol Type:   interface
// Babel Version: 0.10.12
// Release:       $Name$
// Revision:      @(#) $Id$
// Description:   Client-side glue code for sidl.rmi.Invocation
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
// xml-url       = /home/painter/babel-0.10.12/bin/.././share/repository/sidl.rmi.Invocation-v0.9.3.xml
// 

#ifndef included_sidl_rmi_Invocation_hh
#include "sidl_rmi_Invocation.hh"
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
#ifndef included_sidl_io_IOException_hh
#include "sidl_io_IOException.hh"
#endif
#ifndef included_sidl_rmi_NetworkException_hh
#include "sidl_rmi_NetworkException.hh"
#endif
#ifndef included_sidl_rmi_Response_hh
#include "sidl_rmi_Response.hh"
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
ucxx::sidl::rmi::Invocation::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
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
ucxx::sidl::rmi::Invocation::queryInt( /* in */const ::std::string& name )
throw ()

{
  ::ucxx::sidl::BaseInterface _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
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
ucxx::sidl::rmi::Invocation::isType( /* in */const ::std::string& name )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
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
ucxx::sidl::rmi::Invocation::getClassInfo(  )
throw ()

{
  ::ucxx::sidl::ClassInfo _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::ClassInfo( 
    (*(loc_self->d_epv->f_getClassInfo))(loc_self->d_object ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packBool( /* in */const ::std::string& key,
  /* in */bool value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_bool _local_value = value;
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packBool))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ _local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packChar( /* in */const ::std::string& key,
  /* in */char value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packChar))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packInt( /* in */const ::std::string& key,
  /* in */int32_t value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packInt))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packLong( /* in */const ::std::string& key,
  /* in */int64_t value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packLong))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packFloat( /* in */const ::std::string& key,
  /* in */float value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packFloat))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packDouble( /* in */const ::std::string& key,
  /* in */double value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packDouble))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packFcomplex( /* in */const ::std::string& key,
  /* in */const ::std::complex<float>& value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  struct sidl_fcomplex _local_value = {value.real(), value.imag() } ; 
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packFcomplex))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ _local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packDcomplex( /* in */const ::std::string& key,
  /* in */const ::std::complex<double>& value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  struct sidl_dcomplex _local_value = {value.real(), value.imag() } ; 
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packDcomplex))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ _local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
ucxx::sidl::rmi::Invocation::packString( /* in */const ::std::string& key,
  /* in */const ::std::string& value )
throw ( ::ucxx::sidl::io::IOException)

{

  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packString))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ value.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    void * _p = 0;
    if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.io.IOException")) != 0 ) {
      struct sidl_io_IOException__object * _realtype = reinterpret_cast< struct 
        sidl_io_IOException__object*>(_p);
      // Note: alternate constructor does not increment refcount.
      throw ::ucxx::sidl::io::IOException( _realtype, false );
    }
  }
  /*unpack results and cleanup*/
}



/**
 * this method may be called only once at the end of the object's lifetime 
 */
::ucxx::sidl::rmi::Response
ucxx::sidl::rmi::Invocation::invokeMethod(  )
throw ( ::ucxx::sidl::rmi::NetworkException)

{
  ::ucxx::sidl::rmi::Response _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "sidl.rmi.Invocation");
  sidl_BaseInterface__object * _exception = 0;
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::rmi::Response( 
    (*(loc_self->d_epv->f_invokeMethod))(loc_self->d_object, &_exception ),
    false);
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



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// copy constructor
ucxx::sidl::rmi::Invocation::Invocation ( const ::ucxx::sidl::rmi::Invocation& 
  original ) {
  d_self = original._cast("sidl.rmi.Invocation");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::sidl::rmi::Invocation&
ucxx::sidl::rmi::Invocation::operator=( const ::ucxx::sidl::rmi::Invocation& 
  rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("sidl.rmi.Invocation");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::sidl::rmi::Invocation::Invocation ( ::ucxx::sidl::rmi::Invocation::ior_t* 
  ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::sidl::rmi::Invocation::Invocation ( ::ucxx::sidl::rmi::Invocation::ior_t* 
  ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::sidl::rmi::Invocation::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< 
      void*>((*loc_self->d_epv->f__cast)(loc_self->d_object, type));
  }
  return ptr;
}

