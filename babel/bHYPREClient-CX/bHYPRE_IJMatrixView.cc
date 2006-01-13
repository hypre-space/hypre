// 
// File:          bHYPRE_IJMatrixView.cc
// Symbol:        bHYPRE.IJMatrixView-v1.0.0
// Symbol Type:   interface
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.IJMatrixView
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_IJMatrixView_hh
#include "bHYPRE_IJMatrixView.hh"
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
ucxx::bHYPRE::IJMatrixView::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
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
ucxx::bHYPRE::IJMatrixView::queryInt( /* in */const ::std::string& name )
throw ()

{
  ::ucxx::sidl::BaseInterface _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
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
ucxx::bHYPRE::IJMatrixView::isType( /* in */const ::std::string& name )
throw ()

{
  bool _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
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
ucxx::bHYPRE::IJMatrixView::getClassInfo(  )
throw ()

{
  ::ucxx::sidl::ClassInfo _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
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
ucxx::bHYPRE::IJMatrixView::SetCommunicator( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetCommunicator))(loc_self->d_object,
    /* in */ _local_mpi_comm );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::SetValues( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* ncols, /* in rarray[nrows] */int32_t* rows,
  /* in rarray[nnonzeros] */int32_t* cols,
  /* in rarray[nnonzeros] */double* values, /* in */int32_t nnonzeros )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_int__array cols_real;
  struct sidl_int__array *cols_tmp = &cols_real;
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self->d_object,
    /* in rarray[nrows] */ ncols_tmp, /* in rarray[nrows] */ rows_tmp,
    /* in rarray[nnonzeros] */ cols_tmp,
    /* in rarray[nnonzeros] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::SetValues( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> ncols,
  /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self->d_object,
    /* in rarray[nrows] */ ncols._get_ior(),
    /* in rarray[nrows] */ rows._get_ior(),
    /* in rarray[nnonzeros] */ cols._get_ior(),
    /* in rarray[nnonzeros] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::AddToValues( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* ncols, /* in rarray[nrows] */int32_t* rows,
  /* in rarray[nnonzeros] */int32_t* cols,
  /* in rarray[nnonzeros] */double* values, /* in */int32_t nnonzeros )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_int__array cols_real;
  struct sidl_int__array *cols_tmp = &cols_real;
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self->d_object,
    /* in rarray[nrows] */ ncols_tmp, /* in rarray[nrows] */ rows_tmp,
    /* in rarray[nnonzeros] */ cols_tmp,
    /* in rarray[nnonzeros] */ values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::AddToValues( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> ncols,
  /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self->d_object,
    /* in rarray[nrows] */ ncols._get_ior(),
    /* in rarray[nrows] */ rows._get_ior(),
    /* in rarray[nnonzeros] */ cols._get_ior(),
    /* in rarray[nnonzeros] */ values._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::GetRowCounts( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* rows, /* inout rarray[nrows] */int32_t* ncols )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetRowCounts))(loc_self->d_object,
    /* in rarray[nrows] */ rows_tmp, /* inout rarray[nrows] */ &ncols_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::GetRowCounts( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> rows,
  /* inout rarray[nrows] */::ucxx::sidl::array<int32_t>& ncols )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  if (ncols) {
    ncols.addRef();
  }
  struct sidl_int__array* _local_ncols = ncols._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetRowCounts))(loc_self->d_object,
    /* in rarray[nrows] */ rows._get_ior(),
    /* inout rarray[nrows] */ &_local_ncols );
  /*dispatch to ior*/
  ncols._set_ior(_local_ncols);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::GetValues( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* ncols, /* in rarray[nrows] */int32_t* rows,
  /* in rarray[nnonzeros] */int32_t* cols,
  /* inout rarray[nnonzeros] */double* values, /* in */int32_t nnonzeros )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper,
    ncols_stride);
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_int__array cols_real;
  struct sidl_int__array *cols_tmp = &cols_real;
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetValues))(loc_self->d_object,
    /* in rarray[nrows] */ ncols_tmp, /* in rarray[nrows] */ rows_tmp,
    /* in rarray[nnonzeros] */ cols_tmp,
    /* inout rarray[nnonzeros] */ &values_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::GetValues( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> ncols,
  /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
  /* inout rarray[nnonzeros] */::ucxx::sidl::array<double>& values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  if (values) {
    values.addRef();
  }
  struct sidl_double__array* _local_values = values._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetValues))(loc_self->d_object,
    /* in rarray[nrows] */ ncols._get_ior(),
    /* in rarray[nrows] */ rows._get_ior(),
    /* in rarray[nnonzeros] */ cols._get_ior(),
    /* inout rarray[nnonzeros] */ &_local_values );
  /*dispatch to ior*/
  values._set_ior(_local_values);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::SetRowSizes( /* in rarray[nrows] */int32_t* sizes,
  /* in */int32_t nrows )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  int32_t sizes_lower[1], sizes_upper[1], sizes_stride[1];
  struct sidl_int__array sizes_real;
  struct sidl_int__array *sizes_tmp = &sizes_real;
  sizes_upper[0] = nrows-1;
  sidl_int__array_init(sizes, sizes_tmp, 1, sizes_lower, sizes_upper,
    sizes_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetRowSizes))(loc_self->d_object,
    /* in rarray[nrows] */ sizes_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::SetRowSizes( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> sizes )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetRowSizes))(loc_self->d_object,
    /* in rarray[nrows] */ sizes._get_ior() );
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
ucxx::bHYPRE::IJMatrixView::Print( /* in */const ::std::string& filename )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Print))(loc_self->d_object,
    /* in */ filename.c_str() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
ucxx::bHYPRE::IJMatrixView::Read( /* in */const ::std::string& filename,
  /* in */::ucxx::bHYPRE::MPICommunicator comm )
throw ()

{
  int32_t _result;
  ior_t* loc_self = (ior_t*)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > 
    (d_self))->d_epv->f__cast)((reinterpret_cast< struct 
    sidl_BaseInterface__object * > (d_self))->d_object, "bHYPRE.IJMatrixView");
  struct bHYPRE_MPICommunicator__object* _local_comm = comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Read))(loc_self->d_object,
    /* in */ filename.c_str(), /* in */ _local_comm );
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
ucxx::bHYPRE::IJMatrixView::IJMatrixView ( const ::ucxx::bHYPRE::IJMatrixView& 
  original ) {
  d_self = original._cast("bHYPRE.IJMatrixView");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::IJMatrixView&
ucxx::bHYPRE::IJMatrixView::operator=( const ::ucxx::bHYPRE::IJMatrixView& rhs 
  ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.IJMatrixView");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::IJMatrixView::IJMatrixView ( ::ucxx::bHYPRE::IJMatrixView::ior_t* 
  ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::IJMatrixView::IJMatrixView ( ::ucxx::bHYPRE::IJMatrixView::ior_t* 
  ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::IJMatrixView::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< 
      void*>((*loc_self->d_epv->f__cast)(loc_self->d_object, type));
  }
  return ptr;
}

