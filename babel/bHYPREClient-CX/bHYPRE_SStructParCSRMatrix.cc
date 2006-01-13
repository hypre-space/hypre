// 
// File:          bHYPRE_SStructParCSRMatrix.cc
// Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.SStructParCSRMatrix
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_SStructParCSRMatrix_hh
#include "bHYPRE_SStructParCSRMatrix.hh"
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
#ifndef included_bHYPRE_SStructGraph_hh
#include "bHYPRE_SStructGraph.hh"
#endif
#ifndef included_bHYPRE_SStructParCSRMatrix_hh
#include "bHYPRE_SStructParCSRMatrix.hh"
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
::ucxx::bHYPRE::SStructParCSRMatrix
ucxx::bHYPRE::SStructParCSRMatrix::Create( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm,
  /* in */::ucxx::bHYPRE::SStructGraph graph )
throw ()

{
  ::ucxx::bHYPRE::SStructParCSRMatrix _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  struct bHYPRE_SStructGraph__object* _local_graph = graph._get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::SStructParCSRMatrix( ( _get_sepv()->f_Create)( /* 
    in */ _local_mpi_comm, /* in */ _local_graph ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::bHYPRE::SStructParCSRMatrix::isSame( /* in */::ucxx::sidl::BaseInterface 
  iobj )
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
ucxx::bHYPRE::SStructParCSRMatrix::queryInt( /* in */const ::std::string& name )
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
ucxx::bHYPRE::SStructParCSRMatrix::isType( /* in */const ::std::string& name )
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
ucxx::bHYPRE::SStructParCSRMatrix::getClassInfo(  )
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
ucxx::bHYPRE::SStructParCSRMatrix::SetCommunicator( /* in 
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
 *  A semi-structured matrix or vector contains a Struct or IJ matrix
 *  or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::GetObject( /* out 
  */::ucxx::sidl::BaseInterface& A )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct sidl_BaseInterface__object* _local_A;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetObject))(loc_self, /* out */ &_local_A );
  /*dispatch to ior*/
  if ( A._not_nil() ) {
    A.deleteRef();
  }
  A._set_ior( _local_A);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the matrix graph.
 * DEPRECATED     Use Create
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetGraph( /* in 
  */::ucxx::bHYPRE::SStructGraph graph )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_SStructGraph__object* _local_graph = graph._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetGraph))(loc_self, /* in */ _local_graph );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* index, /* in */int32_t dim, /* in */int32_t var,
  /* in */int32_t nentries, /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nentries] */double* values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t index_lower[1], index_upper[1], index_stride[1];
  struct sidl_int__array index_real;
  struct sidl_int__array *index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nentries-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ index_tmp, /* in */ var,
    /* in rarray[nentries] */ entries_tmp,
    /* in rarray[nentries] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetValues( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> index, /* in */int32_t var,
  /* in rarray[nentries] */::ucxx::sidl::array<int32_t> entries,
  /* in rarray[nentries] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ index._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nentries] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetBoxValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* ilower, /* in rarray[dim] */int32_t* iupper,
  /* in */int32_t dim, /* in */int32_t var, /* in */int32_t nentries,
  /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nvalues] */double* values, /* in */int32_t nvalues )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
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
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetBoxValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower_tmp, /* in rarray[dim] */ iupper_tmp,
    /* in */ var, /* in rarray[nentries] */ entries_tmp,
    /* in rarray[nvalues] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetBoxValues( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> ilower,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper, /* in */int32_t var,
  /* in rarray[nentries] */::ucxx::sidl::array<int32_t> entries,
  /* in rarray[nvalues] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetBoxValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower._get_ior(),
    /* in rarray[dim] */ iupper._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nvalues] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::AddToValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* index, /* in */int32_t dim, /* in */int32_t var,
  /* in */int32_t nentries, /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nentries] */double* values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t index_lower[1], index_upper[1], index_stride[1];
  struct sidl_int__array index_real;
  struct sidl_int__array *index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nentries-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ index_tmp, /* in */ var,
    /* in rarray[nentries] */ entries_tmp,
    /* in rarray[nentries] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::AddToValues( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> index, /* in */int32_t var,
  /* in rarray[nentries] */::ucxx::sidl::array<int32_t> entries,
  /* in rarray[nentries] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ index._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nentries] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::AddToBoxValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* ilower, /* in rarray[dim] */int32_t* iupper,
  /* in */int32_t dim, /* in */int32_t var, /* in */int32_t nentries,
  /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nvalues] */double* values, /* in */int32_t nvalues )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
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
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToBoxValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower_tmp, /* in rarray[dim] */ iupper_tmp,
    /* in */ var, /* in rarray[nentries] */ entries_tmp,
    /* in rarray[nvalues] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::AddToBoxValues( /* in */int32_t part,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> ilower,
  /* in rarray[dim] */::ucxx::sidl::array<int32_t> iupper, /* in */int32_t var,
  /* in rarray[nentries] */::ucxx::sidl::array<int32_t> entries,
  /* in rarray[nvalues] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToBoxValues))(loc_self, /* in */ part,
    /* in rarray[dim] */ ilower._get_ior(),
    /* in rarray[dim] */ iupper._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nvalues] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::Print( /* in */const ::std::string& filename,
  /* in */int32_t all )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Print))(loc_self, /* in */ filename.c_str(),
    /* in */ all );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetIntParameter( /* in */const 
  ::std::string& name, /* in */int32_t value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntParameter))(loc_self,
    /* in */ name.c_str(), /* in */ value );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetDoubleParameter( /* in */const 
  ::std::string& name, /* in */double value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleParameter))(loc_self,
    /* in */ name.c_str(), /* in */ value );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the string parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetStringParameter( /* in */const 
  ::std::string& name, /* in */const ::std::string& value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetStringParameter))(loc_self,
    /* in */ name.c_str(), /* in */ value.c_str() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetIntArray1Parameter( /* in */const 
  ::std::string& name, /* in rarray[nvalues] */int32_t* value,
  /* in */int32_t nvalues )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t value_lower[1], value_upper[1], value_stride[1];
  struct sidl_int__array value_real;
  struct sidl_int__array *value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_int__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntArray1Parameter))(loc_self,
    /* in */ name.c_str(), /* in rarray[nvalues] */ value_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetIntArray1Parameter( /* in */const 
  ::std::string& name,
  /* in rarray[nvalues] */::ucxx::sidl::array<int32_t> value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntArray1Parameter))(loc_self,
    /* in */ name.c_str(), /* in rarray[nvalues] */ value._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetIntArray2Parameter( /* in */const 
  ::std::string& name, /* in array<int,2,
  column-major> */::ucxx::sidl::array<int32_t> value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntArray2Parameter))(loc_self,
    /* in */ name.c_str(), /* in array<int,2,
    column-major> */ value._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetDoubleArray1Parameter( /* in */const 
  ::std::string& name, /* in rarray[nvalues] */double* value,
  /* in */int32_t nvalues )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t value_lower[1], value_upper[1], value_stride[1];
  struct sidl_double__array value_real;
  struct sidl_double__array *value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_double__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleArray1Parameter))(loc_self,
    /* in */ name.c_str(), /* in rarray[nvalues] */ value_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetDoubleArray1Parameter( /* in */const 
  ::std::string& name,
  /* in rarray[nvalues] */::ucxx::sidl::array<double> value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleArray1Parameter))(loc_self,
    /* in */ name.c_str(), /* in rarray[nvalues] */ value._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::SetDoubleArray2Parameter( /* in */const 
  ::std::string& name, /* in array<double,2,
  column-major> */::ucxx::sidl::array<double> value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleArray2Parameter))(loc_self,
    /* in */ name.c_str(), /* in array<double,2,
    column-major> */ value._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::GetIntValue( /* in */const ::std::string& 
  name, /* out */int32_t& value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetIntValue))(loc_self, /* in */ name.c_str(),
    /* out */ &value );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Get the double parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::GetDoubleValue( /* in */const ::std::string& 
  name, /* out */double& value )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetDoubleValue))(loc_self,
    /* in */ name.c_str(), /* out */ &value );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::Setup( /* in */::ucxx::bHYPRE::Vector b,
  /* in */::ucxx::bHYPRE::Vector x )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_Vector__object* _local_b = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( b._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (b._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (b._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  struct bHYPRE_Vector__object* _local_x = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( x._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (x._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (x._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Setup))(loc_self, /* in */ _local_b,
    /* in */ _local_x );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::Apply( /* in */::ucxx::bHYPRE::Vector b,
  /* inout */::ucxx::bHYPRE::Vector& x )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_Vector__object* _local_b = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( b._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (b._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (b._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  struct bHYPRE_Vector__object* _local_x = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( x._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (x._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (x._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  x._set_ior( 0 );
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Apply))(loc_self, /* in */ _local_b,
    /* inout */ &_local_x );
  /*dispatch to ior*/
  x._set_ior( _local_x);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */
int32_t
ucxx::bHYPRE::SStructParCSRMatrix::ApplyAdjoint( /* in */::ucxx::bHYPRE::Vector 
  b, /* inout */::ucxx::bHYPRE::Vector& x )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct bHYPRE_Vector__object* _local_b = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( b._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (b._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (b._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  struct bHYPRE_Vector__object* _local_x = reinterpret_cast< struct 
    bHYPRE_Vector__object* > ( x._get_ior() ? ((*((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (x._get_ior()))->d_epv->f__cast))((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (x._get_ior()))->d_object,
    "bHYPRE.Vector")) : 0);
  x._set_ior( 0 );
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_ApplyAdjoint))(loc_self, /* in */ _local_b,
    /* inout */ &_local_x );
  /*dispatch to ior*/
  x._set_ior( _local_x);
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
::ucxx::bHYPRE::SStructParCSRMatrix
ucxx::bHYPRE::SStructParCSRMatrix::_create() {
  ::ucxx::bHYPRE::SStructParCSRMatrix self( (*_get_ext()->createObject)(),
    false );
  return self;
}

// copy constructor
ucxx::bHYPRE::SStructParCSRMatrix::SStructParCSRMatrix ( const 
  ::ucxx::bHYPRE::SStructParCSRMatrix& original ) {
  d_self = original._cast("bHYPRE.SStructParCSRMatrix");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::SStructParCSRMatrix&
ucxx::bHYPRE::SStructParCSRMatrix::operator=( const 
  ::ucxx::bHYPRE::SStructParCSRMatrix& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.SStructParCSRMatrix");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::SStructParCSRMatrix::SStructParCSRMatrix ( 
  ::ucxx::bHYPRE::SStructParCSRMatrix::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::SStructParCSRMatrix::SStructParCSRMatrix ( 
  ::ucxx::bHYPRE::SStructParCSRMatrix::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::SStructParCSRMatrix::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::bHYPRE::SStructParCSRMatrix::ext_t * 
  ucxx::bHYPRE::SStructParCSRMatrix::s_ext = 0;

// private static method to get static data type
const ::ucxx::bHYPRE::SStructParCSRMatrix::ext_t *
ucxx::bHYPRE::SStructParCSRMatrix::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = bHYPRE_SStructParCSRMatrix__externals();
#else
    s_ext = (struct 
      bHYPRE_SStructParCSRMatrix__external*)sidl_dynamicLoadIOR(
      "bHYPRE.SStructParCSRMatrix","bHYPRE_SStructParCSRMatrix__externals") ;
#endif
  }
  return s_ext;
}

