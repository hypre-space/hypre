// 
// File:          bHYPRE_SStructStencil.cc
// Symbol:        bHYPRE.SStructStencil-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.SStructStencil
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_SStructStencil_hh
#include "bHYPRE_SStructStencil.hh"
#endif
#ifndef included_sidl_BaseInterface_hh
#include "sidl_BaseInterface.hh"
#endif
#ifndef included_sidl_BaseClass_hh
#include "sidl_BaseClass.hh"
#endif
#include "sidl_String.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.hh"
#include "sidl_DLL.hh"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_bHYPRE_SStructStencil_hh
#include "bHYPRE_SStructStencil.hh"
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
 * user defined static method
 */
::ucxx::bHYPRE::SStructStencil
ucxx::bHYPRE::SStructStencil::Create( /* in */int32_t ndim,
  /* in */int32_t size )
throw ()

{
  ::ucxx::bHYPRE::SStructStencil _result;
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::SStructStencil( ( _get_sepv()->f_Create)( /* in */ 
    ndim, /* in */ size ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::bHYPRE::SStructStencil::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
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
ucxx::bHYPRE::SStructStencil::queryInt( /* in */const ::std::string& name )
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
ucxx::bHYPRE::SStructStencil::isType( /* in */const ::std::string& name )
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
ucxx::bHYPRE::SStructStencil::getClassInfo(  )
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
 * Set a stencil entry.
 * 
 */
int32_t
ucxx::bHYPRE::SStructStencil::SetEntry( /* in */int32_t entry,
  /* in rarray[dim] */int32_t* offset, /* in */int32_t dim,
  /* in */int32_t var )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t offset_lower[1], offset_upper[1], offset_stride[1];
  struct sidl_int__array offset_real;
  struct sidl_int__array *offset_tmp = &offset_real;
  offset_upper[0] = dim-1;
  sidl_int__array_init(offset, offset_tmp, 1, offset_lower, offset_upper,
    offset_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetEntry))(loc_self, /* in */ entry,
    /* in rarray[dim] */ offset_tmp, /* in */ var );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set a stencil entry.
 * 
 */
int32_t
ucxx::bHYPRE::SStructStencil::SetEntry( /* in */int32_t entry,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> offset, /* in */int32_t var )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetEntry))(loc_self, /* in */ entry,
    /* in rarray[dim] */ offset._get_ior(), /* in */ var );
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
::ucxx::bHYPRE::SStructStencil
ucxx::bHYPRE::SStructStencil::_create() {
  ::ucxx::bHYPRE::SStructStencil self( (*_get_ext()->createObject)(), false );
  return self;
}

// copy constructor
ucxx::bHYPRE::SStructStencil::SStructStencil ( const 
  ::ucxx::bHYPRE::SStructStencil& original ) {
  d_self = original._cast("bHYPRE.SStructStencil");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::SStructStencil&
ucxx::bHYPRE::SStructStencil::operator=( const ::ucxx::bHYPRE::SStructStencil& 
  rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.SStructStencil");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::SStructStencil::SStructStencil ( 
  ::ucxx::bHYPRE::SStructStencil::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::SStructStencil::SStructStencil ( 
  ::ucxx::bHYPRE::SStructStencil::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::SStructStencil::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::bHYPRE::SStructStencil::ext_t * 
  ucxx::bHYPRE::SStructStencil::s_ext = 0;

// private static method to get static data type
const ::ucxx::bHYPRE::SStructStencil::ext_t *
ucxx::bHYPRE::SStructStencil::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = bHYPRE_SStructStencil__externals();
#else
    s_ext = (struct 
      bHYPRE_SStructStencil__external*)sidl_dynamicLoadIOR(
      "bHYPRE.SStructStencil","bHYPRE_SStructStencil__externals") ;
#endif
  }
  return s_ext;
}

