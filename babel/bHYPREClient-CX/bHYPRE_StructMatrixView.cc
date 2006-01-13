// 
// File:          bHYPRE_StructMatrixView.cc
// Symbol:        bHYPRE.StructMatrixView-v1.0.0
// Symbol Type:   interface
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.StructMatrixView
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_StructMatrixView_hh
#include "bHYPRE_StructMatrixView.hh"
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
#ifndef included_bHYPRE_StructGrid_hh
#include "bHYPRE_StructGrid.hh"
#endif
#ifndef included_bHYPRE_StructStencil_hh
#include "bHYPRE_StructStencil.hh"
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
ucxx::bHYPRE::StructMatrixView::isSame( /* in */::ucxx::sidl::BaseInterface 
  iobj )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
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
ucxx::bHYPRE::StructMatrixView::queryInt( /* in */const ::std::string& name )
throw ()

{
  ::ucxx::sidl::BaseInterface _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
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
ucxx::bHYPRE::StructMatrixView::isType( /* in */const ::std::string& name )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
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
ucxx::bHYPRE::StructMatrixView::getClassInfo(  )
throw ()

{
  ::ucxx::sidl::ClassInfo _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
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
ucxx::bHYPRE::StructMatrixView::SetCommunicator( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetCommunicator))(loc_self->d_object,
    /* in */ _local_mpi_comm );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetGrid( /* in */::ucxx::bHYPRE::StructGrid 
  grid )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  struct bHYPRE_StructGrid__object* _local_grid = grid._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetGrid))(loc_self->d_object,
    /* in */ _local_grid );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetStencil( /* in 
  */::ucxx::bHYPRE::StructStencil stencil )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  struct bHYPRE_StructStencil__object* _local_stencil = stencil._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetStencil))(loc_self->d_object,
    /* in */ _local_stencil );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetValues( /* in rarray[dim] */int32_t* index,
  /* in */int32_t dim, /* in */int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */double* values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  int32_t index_lower[1], index_upper[1], index_stride[1];
  struct sidl_int__array index_real;
  struct sidl_int__array *index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1];
  struct sidl_int__array stencil_indices_real;
  struct sidl_int__array *stencil_indices_tmp = &stencil_indices_real;
  stencil_indices_upper[0] = num_stencil_indices-1;
  sidl_int__array_init(stencil_indices, stencil_indices_tmp, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = num_stencil_indices-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self->d_object,
    /* in rarray[dim] */ index_tmp,
    /* in rarray[num_stencil_indices] */ stencil_indices_tmp,
    /* in rarray[num_stencil_indices] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetValues( /* in rarray[dim] 
  */::ucxx::sidl::array<int32_t> index,
  /* in rarray[num_stencil_indices] */::ucxx::sidl::array<int32_t> 
  stencil_indices,
  /* in rarray[num_stencil_indices] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self->d_object,
    /* in rarray[dim] */ index._get_ior(),
    /* in rarray[num_stencil_indices] */ stencil_indices._get_ior(),
    /* in rarray[num_stencil_indices] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetBoxValues( /* in rarray[dim] */int32_t* 
  ilower, /* in rarray[dim] */int32_t* iupper, /* in */int32_t dim,
  /* in */int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */int32_t* stencil_indices,
  /* in rarray[nvalues] */double* values, /* in */int32_t nvalues )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1];
  struct sidl_int__array ilower_real;
  struct sidl_int__array *ilower_tmp = &ilower_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1];
  struct sidl_int__array iupper_real;
  struct sidl_int__array *iupper_tmp = &iupper_real;
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1];
  struct sidl_int__array stencil_indices_real;
  struct sidl_int__array *stencil_indices_tmp = &stencil_indices_real;
  stencil_indices_upper[0] = num_stencil_indices-1;
  sidl_int__array_init(stencil_indices, stencil_indices_tmp, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetBoxValues))(loc_self->d_object,
    /* in rarray[dim] */ ilower_tmp, /* in rarray[dim] */ iupper_tmp,
    /* in rarray[num_stencil_indices] */ stencil_indices_tmp,
    /* in rarray[nvalues] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetBoxValues( /* in rarray[dim] 
  */::ucxx::sidl::array<int32_t> ilower,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper,
  /* in rarray[num_stencil_indices] */::ucxx::sidl::array<int32_t> 
  stencil_indices, /* in rarray[nvalues] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetBoxValues))(loc_self->d_object,
    /* in rarray[dim] */ ilower._get_ior(),
    /* in rarray[dim] */ iupper._get_ior(),
    /* in rarray[num_stencil_indices] */ stencil_indices._get_ior(),
    /* in rarray[nvalues] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetNumGhost( /* in rarray[dim2] */int32_t* 
  num_ghost, /* in */int32_t dim2 )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1];
  struct sidl_int__array num_ghost_real;
  struct sidl_int__array *num_ghost_tmp = &num_ghost_real;
  num_ghost_upper[0] = dim2-1;
  sidl_int__array_init(num_ghost, num_ghost_tmp, 1, num_ghost_lower,
    num_ghost_upper, num_ghost_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetNumGhost))(loc_self->d_object,
    /* in rarray[dim2] */ num_ghost_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetNumGhost( /* in rarray[dim2] 
  */::ucxx::sidl::array<int32_t> num_ghost )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetNumGhost))(loc_self->d_object,
    /* in rarray[dim2] */ num_ghost._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetConstantEntries( /* in */int32_t 
  num_stencil_constant_points,
  /* in rarray[num_stencil_constant_points] */int32_t* stencil_constant_points )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  int32_t stencil_constant_points_lower[1], stencil_constant_points_upper[1],
    stencil_constant_points_stride[1];
  struct sidl_int__array stencil_constant_points_real;
  struct sidl_int__array *stencil_constant_points_tmp = 
    &stencil_constant_points_real;
  stencil_constant_points_upper[0] = num_stencil_constant_points-1;
  sidl_int__array_init(stencil_constant_points, stencil_constant_points_tmp, 1,
    stencil_constant_points_lower, stencil_constant_points_upper,
    stencil_constant_points_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetConstantEntries))(loc_self->d_object,
    /* in rarray[num_stencil_constant_points] */ stencil_constant_points_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetConstantEntries( /* in 
  rarray[num_stencil_constant_points] */::ucxx::sidl::array<int32_t> 
  stencil_constant_points )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetConstantEntries))(loc_self->d_object,
    /* in rarray[num_stencil_constant_points] */ 
    stencil_constant_points._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method.
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetConstantValues( /* in */int32_t 
  num_stencil_indices,
  /* in rarray[num_stencil_indices] */int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */double* values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1];
  struct sidl_int__array stencil_indices_real;
  struct sidl_int__array *stencil_indices_tmp = &stencil_indices_real;
  stencil_indices_upper[0] = num_stencil_indices-1;
  sidl_int__array_init(stencil_indices, stencil_indices_tmp, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = num_stencil_indices-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetConstantValues))(loc_self->d_object,
    /* in rarray[num_stencil_indices] */ stencil_indices_tmp,
    /* in rarray[num_stencil_indices] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined non-static method
 */
int32_t
ucxx::bHYPRE::StructMatrixView::SetConstantValues( /* in 
  rarray[num_stencil_indices] */::ucxx::sidl::array<int32_t> stencil_indices,
  /* in rarray[num_stencil_indices] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object,
    "bHYPRE.StructMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetConstantValues))(loc_self->d_object,
    /* in rarray[num_stencil_indices] */ stencil_indices._get_ior(),
    /* in rarray[num_stencil_indices] */ values._get_ior() );
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

// copy constructor
ucxx::bHYPRE::StructMatrixView::StructMatrixView ( const 
  ::ucxx::bHYPRE::StructMatrixView& original ) {
  d_self = original._cast("bHYPRE.StructMatrixView");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::StructMatrixView&
ucxx::bHYPRE::StructMatrixView::operator=( const 
  ::ucxx::bHYPRE::StructMatrixView& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.StructMatrixView");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::StructMatrixView::StructMatrixView ( 
  ::ucxx::bHYPRE::StructMatrixView::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::StructMatrixView::StructMatrixView ( 
  ::ucxx::bHYPRE::StructMatrixView::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::StructMatrixView::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< 
      void*>((*loc_self->d_epv->f__cast)(loc_self->d_object, type));
  }
  return ptr;
}

