// 
// File:          bHYPRE_SStructMatrixVectorView.cc
// Symbol:        bHYPRE.SStructMatrixVectorView-v1.0.0
// Symbol Type:   interface
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.SStructMatrixVectorView
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_SStructMatrixVectorView_hh
#include "bHYPRE_SStructMatrixVectorView.hh"
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
#ifndef included_bHYPRE_MPICommunicator_hh
#include "bHYPRE_MPICommunicator.hh"
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
ucxx::bHYPRE::SStructMatrixVectorView::isSame( /* in 
  */::ucxx::sidl::BaseInterface iobj )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.SStructMatrixVectorView");
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
ucxx::bHYPRE::SStructMatrixVectorView::queryInt( /* in */const ::std::string& 
  name )
throw ()

{
  ::ucxx::sidl::BaseInterface _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.SStructMatrixVectorView");
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
ucxx::bHYPRE::SStructMatrixVectorView::isType( /* in */const ::std::string& 
  name )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.SStructMatrixVectorView");
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
ucxx::bHYPRE::SStructMatrixVectorView::getClassInfo(  )
throw ()

{
  ::ucxx::sidl::ClassInfo _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.SStructMatrixVectorView");
  /*pack args to dispatch to ior*/
  _result = ::ucxx::sidl::ClassInfo( 
    (*(loc_self->d_epv->f_getClassInfo))(loc_self->d_object ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
ucxx::bHYPRE::SStructMatrixVectorView::SetCommunicator( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.SStructMatrixVectorView");
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetCommunicator))(loc_self->d_object,
    /* in */ _local_mpi_comm );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 *  A semi-structured matrix or vector contains a Struct or IJ matrix
 *  or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
ucxx::bHYPRE::SStructMatrixVectorView::GetObject( /* out 
  */::ucxx::sidl::BaseInterface& A )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.SStructMatrixVectorView");
  struct sidl_BaseInterface__object* _local_A;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetObject))(loc_self->d_object,
    /* out */ &_local_A );
  /*dispatch to ior*/
  if ( A._not_nil() ) {
    A.deleteRef();
  }
  A._set_ior( _local_A);
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
ucxx::bHYPRE::SStructMatrixVectorView::SStructMatrixVectorView ( const 
  ::ucxx::bHYPRE::SStructMatrixVectorView& original ) {
  d_self = original._cast("bHYPRE.SStructMatrixVectorView");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::SStructMatrixVectorView&
ucxx::bHYPRE::SStructMatrixVectorView::operator=( const 
  ::ucxx::bHYPRE::SStructMatrixVectorView& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.SStructMatrixVectorView");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::SStructMatrixVectorView::SStructMatrixVectorView ( 
  ::ucxx::bHYPRE::SStructMatrixVectorView::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::SStructMatrixVectorView::SStructMatrixVectorView ( 
  ::ucxx::bHYPRE::SStructMatrixVectorView::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::SStructMatrixVectorView::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< 
      void*>((*loc_self->d_epv->f__cast)(loc_self->d_object, type));
  }
  return ptr;
}

