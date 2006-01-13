// 
// File:          bHYPRE_IJParCSRVector.cc
// Symbol:        bHYPRE.IJParCSRVector-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.IJParCSRVector
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_IJParCSRVector_hh
#include "bHYPRE_IJParCSRVector.hh"
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
#ifndef included_bHYPRE_IJParCSRVector_hh
#include "bHYPRE_IJParCSRVector.hh"
#endif
#ifndef included_bHYPRE_MPICommunicator_hh
#include "bHYPRE_MPICommunicator.hh"
#endif
#ifndef included_bHYPRE_Vector_hh
#include "bHYPRE_Vector.hh"
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
::ucxx::bHYPRE::IJParCSRVector
ucxx::bHYPRE::IJParCSRVector::Create( /* in */::ucxx::bHYPRE::MPICommunicator 
  mpi_comm, /* in */int32_t jlower, /* in */int32_t jupper )
throw ()

{
  ::ucxx::bHYPRE::IJParCSRVector _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::IJParCSRVector( ( _get_sepv()->f_Create)( /* in */ 
    _local_mpi_comm, /* in */ jlower, /* in */ jupper ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::bHYPRE::IJParCSRVector::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
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
ucxx::bHYPRE::IJParCSRVector::queryInt( /* in */const ::std::string& name )
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
ucxx::bHYPRE::IJParCSRVector::isType( /* in */const ::std::string& name )
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
ucxx::bHYPRE::IJParCSRVector::getClassInfo(  )
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
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::SetCommunicator( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetCommunicator))(loc_self,
    /* in */ _local_mpi_comm );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::SetValues( /* in */int32_t nvalues,
  /* in rarray[nvalues] */int32_t* indices,
  /* in rarray[nvalues] */double* values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t indices_lower[1], indices_upper[1], indices_stride[1];
  struct sidl_int__array indices_real;
  struct sidl_int__array *indices_tmp = &indices_real;
  indices_upper[0] = nvalues-1;
  sidl_int__array_init(indices, indices_tmp, 1, indices_lower, indices_upper,
    indices_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self,
    /* in rarray[nvalues] */ indices_tmp, /* in rarray[nvalues] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::SetValues( /* in rarray[nvalues] 
  */::ucxx::sidl::array<int32_t> indices,
  /* in rarray[nvalues] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self,
    /* in rarray[nvalues] */ indices._get_ior(),
    /* in rarray[nvalues] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::AddToValues( /* in */int32_t nvalues,
  /* in rarray[nvalues] */int32_t* indices,
  /* in rarray[nvalues] */double* values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t indices_lower[1], indices_upper[1], indices_stride[1];
  struct sidl_int__array indices_real;
  struct sidl_int__array *indices_tmp = &indices_real;
  indices_upper[0] = nvalues-1;
  sidl_int__array_init(indices, indices_tmp, 1, indices_lower, indices_upper,
    indices_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self,
    /* in rarray[nvalues] */ indices_tmp, /* in rarray[nvalues] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::AddToValues( /* in rarray[nvalues] 
  */::ucxx::sidl::array<int32_t> indices,
  /* in rarray[nvalues] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self,
    /* in rarray[nvalues] */ indices._get_ior(),
    /* in rarray[nvalues] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::GetValues( /* in */int32_t nvalues,
  /* in rarray[nvalues] */int32_t* indices,
  /* inout rarray[nvalues] */double* values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t indices_lower[1], indices_upper[1], indices_stride[1];
  struct sidl_int__array indices_real;
  struct sidl_int__array *indices_tmp = &indices_real;
  indices_upper[0] = nvalues-1;
  sidl_int__array_init(indices, indices_tmp, 1, indices_lower, indices_upper,
    indices_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetValues))(loc_self,
    /* in rarray[nvalues] */ indices_tmp,
    /* inout rarray[nvalues] */ &values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::GetValues( /* in rarray[nvalues] 
  */::ucxx::sidl::array<int32_t> indices,
  /* inout rarray[nvalues] */::ucxx::sidl::array<double>& values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  if (values) {
    values.addRef();
  }
  struct sidl_double__array* _local_values = values._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetValues))(loc_self,
    /* in rarray[nvalues] */ indices._get_ior(),
    /* inout rarray[nvalues] */ &_local_values );
  /*dispatch to ior*/
  values._set_ior(_local_values);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::Print( /* in */const ::std::string& filename )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Print))(loc_self, /* in */ filename.c_str() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::Read( /* in */const ::std::string& filename,
  /* in */::ucxx::bHYPRE::MPICommunicator comm )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_MPICommunicator__object* _local_comm = comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Read))(loc_self, /* in */ filename.c_str(),
    /* in */ _local_comm );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::Copy( /* in */::ucxx::bHYPRE::Vector x )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_Vector__object* _local_x = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( x._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (x._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (x._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Copy))(loc_self, /* in */ _local_x );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::Clone( /* out */::ucxx::bHYPRE::Vector& x )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_Vector__object* _local_x;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Clone))(loc_self, /* out */ &_local_x );
  /*dispatch to ior*/
  if ( x._not_nil() ) {
    x.deleteRef();
  }
  x._set_ior( _local_x);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::Dot( /* in */::ucxx::bHYPRE::Vector x,
  /* out */double& d )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_Vector__object* _local_x = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( x._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (x._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (x._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Dot))(loc_self, /* in */ _local_x,
    /* out */ &d );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRVector::Axpy( /* in */double a,
  /* in */::ucxx::bHYPRE::Vector x )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_Vector__object* _local_x = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( x._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (x._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (x._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Axpy))(loc_self, /* in */ a,
    /* in */ _local_x );
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
::ucxx::bHYPRE::IJParCSRVector
ucxx::bHYPRE::IJParCSRVector::_create() {
  ::ucxx::bHYPRE::IJParCSRVector self( (*_get_ext()->createObject)(), false );
  return self;
}

// copy constructor
ucxx::bHYPRE::IJParCSRVector::IJParCSRVector ( const 
  ::ucxx::bHYPRE::IJParCSRVector& original ) {
  d_self = original._cast("bHYPRE.IJParCSRVector");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::IJParCSRVector&
ucxx::bHYPRE::IJParCSRVector::operator=( const ::ucxx::bHYPRE::IJParCSRVector& 
  rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.IJParCSRVector");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::IJParCSRVector::IJParCSRVector ( 
  ::ucxx::bHYPRE::IJParCSRVector::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::IJParCSRVector::IJParCSRVector ( 
  ::ucxx::bHYPRE::IJParCSRVector::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::IJParCSRVector::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::bHYPRE::IJParCSRVector::ext_t * 
  ucxx::bHYPRE::IJParCSRVector::s_ext = 0;

// private static method to get static data type
const ::ucxx::bHYPRE::IJParCSRVector::ext_t *
ucxx::bHYPRE::IJParCSRVector::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = bHYPRE_IJParCSRVector__externals();
#else
    s_ext = (struct 
      bHYPRE_IJParCSRVector__external*)sidl_dynamicLoadIOR(
      "bHYPRE.IJParCSRVector","bHYPRE_IJParCSRVector__externals") ;
#endif
  }
  return s_ext;
}

