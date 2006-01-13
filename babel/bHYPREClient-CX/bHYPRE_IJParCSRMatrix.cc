// 
// File:          bHYPRE_IJParCSRMatrix.cc
// Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
// Symbol Type:   class
// Babel Version: 0.10.12
// Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
// 
// WARNING: Automatically generated; changes will be lost
// 
// babel-version = 0.10.12
// 

#ifndef included_bHYPRE_IJParCSRMatrix_hh
#include "bHYPRE_IJParCSRMatrix.hh"
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
#ifndef included_bHYPRE_IJParCSRMatrix_hh
#include "bHYPRE_IJParCSRMatrix.hh"
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
::ucxx::bHYPRE::IJParCSRMatrix
ucxx::bHYPRE::IJParCSRMatrix::Create( /* in */::ucxx::bHYPRE::MPICommunicator 
  mpi_comm, /* in */int32_t ilower, /* in */int32_t iupper,
  /* in */int32_t jlower, /* in */int32_t jupper )
throw ()

{
  ::ucxx::bHYPRE::IJParCSRMatrix _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::IJParCSRMatrix( ( _get_sepv()->f_Create)( /* in */ 
    _local_mpi_comm, /* in */ ilower, /* in */ iupper, /* in */ jlower,
    /* in */ jupper ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined static method
 */
::ucxx::bHYPRE::IJParCSRMatrix
ucxx::bHYPRE::IJParCSRMatrix::GenerateLaplacian( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm, /* in */int32_t nx,
  /* in */int32_t ny, /* in */int32_t nz, /* in */int32_t Px,
  /* in */int32_t Py, /* in */int32_t Pz, /* in */int32_t p, /* in */int32_t q,
  /* in */int32_t r, /* in rarray[nvalues] */double* values,
  /* in */int32_t nvalues, /* in */int32_t discretization )
throw ()

{
  ::ucxx::bHYPRE::IJParCSRMatrix _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::IJParCSRMatrix( ( 
    _get_sepv()->f_GenerateLaplacian)( /* in */ _local_mpi_comm, /* in */ nx,
    /* in */ ny, /* in */ nz, /* in */ Px, /* in */ Py, /* in */ Pz, /* in */ p,
    /* in */ q, /* in */ r, /* in rarray[nvalues] */ values_tmp,
    /* in */ discretization ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined static method
 */
::ucxx::bHYPRE::IJParCSRMatrix
ucxx::bHYPRE::IJParCSRMatrix::GenerateLaplacian( /* in 
  */::ucxx::bHYPRE::MPICommunicator mpi_comm, /* in */int32_t nx,
  /* in */int32_t ny, /* in */int32_t nz, /* in */int32_t Px,
  /* in */int32_t Py, /* in */int32_t Pz, /* in */int32_t p, /* in */int32_t q,
  /* in */int32_t r, /* in rarray[nvalues] */::ucxx::sidl::array<double> values,
  /* in */int32_t discretization )
throw ()

{
  ::ucxx::bHYPRE::IJParCSRMatrix _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = mpi_comm._get_ior();
  /*pack args to dispatch to ior*/
  _result = ::ucxx::bHYPRE::IJParCSRMatrix( ( 
    _get_sepv()->f_GenerateLaplacian)( /* in */ _local_mpi_comm, /* in */ nx,
    /* in */ ny, /* in */ nz, /* in */ Px, /* in */ Py, /* in */ Pz, /* in */ p,
    /* in */ q, /* in */ r, /* in rarray[nvalues] */ values._get_ior(),
    /* in */ discretization ), false);
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
bool
ucxx::bHYPRE::IJParCSRMatrix::isSame( /* in */::ucxx::sidl::BaseInterface iobj )
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
ucxx::bHYPRE::IJParCSRMatrix::queryInt( /* in */const ::std::string& name )
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
ucxx::bHYPRE::IJParCSRMatrix::isType( /* in */const ::std::string& name )
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
ucxx::bHYPRE::IJParCSRMatrix::getClassInfo(  )
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
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRMatrix::SetDiagOffdSizes( /* in rarray[local_nrows] 
  */int32_t* diag_sizes, /* in rarray[local_nrows] */int32_t* offdiag_sizes,
  /* in */int32_t local_nrows )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t diag_sizes_lower[1], diag_sizes_upper[1], diag_sizes_stride[1];
  struct sidl_int__array diag_sizes_real;
  struct sidl_int__array *diag_sizes_tmp = &diag_sizes_real;
  diag_sizes_upper[0] = local_nrows-1;
  sidl_int__array_init(diag_sizes, diag_sizes_tmp, 1, diag_sizes_lower,
    diag_sizes_upper, diag_sizes_stride);
  int32_t offdiag_sizes_lower[1], offdiag_sizes_upper[1],
    offdiag_sizes_stride[1];
  struct sidl_int__array offdiag_sizes_real;
  struct sidl_int__array *offdiag_sizes_tmp = &offdiag_sizes_real;
  offdiag_sizes_upper[0] = local_nrows-1;
  sidl_int__array_init(offdiag_sizes, offdiag_sizes_tmp, 1, offdiag_sizes_lower,
    offdiag_sizes_upper, offdiag_sizes_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDiagOffdSizes))(loc_self,
    /* in rarray[local_nrows] */ diag_sizes_tmp,
    /* in rarray[local_nrows] */ offdiag_sizes_tmp );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRMatrix::SetDiagOffdSizes( /* in rarray[local_nrows] 
  */::ucxx::sidl::array<int32_t> diag_sizes,
  /* in rarray[local_nrows] */::ucxx::sidl::array<int32_t> offdiag_sizes )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDiagOffdSizes))(loc_self,
    /* in rarray[local_nrows] */ diag_sizes._get_ior(),
    /* in rarray[local_nrows] */ offdiag_sizes._get_ior() );
  /*dispatch to ior*/
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRMatrix::SetCommunicator( /* in 
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
ucxx::bHYPRE::IJParCSRMatrix::SetValues( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* ncols, /* in rarray[nrows] */int32_t* rows,
  /* in rarray[nnonzeros] */int32_t* cols,
  /* in rarray[nnonzeros] */double* values, /* in */int32_t nnonzeros )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
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
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::SetValues( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> ncols,
  /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::AddToValues( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* ncols, /* in rarray[nrows] */int32_t* rows,
  /* in rarray[nnonzeros] */int32_t* cols,
  /* in rarray[nnonzeros] */double* values, /* in */int32_t nnonzeros )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
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
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::AddToValues( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> ncols,
  /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<double> values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::GetRowCounts( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* rows, /* inout rarray[nrows] */int32_t* ncols )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
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
  _result = (*(loc_self->d_epv->f_GetRowCounts))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::GetRowCounts( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> rows,
  /* inout rarray[nrows] */::ucxx::sidl::array<int32_t>& ncols )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  if (ncols) {
    ncols.addRef();
  }
  struct sidl_int__array* _local_ncols = ncols._get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetRowCounts))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::GetValues( /* in */int32_t nrows,
  /* in rarray[nrows] */int32_t* ncols, /* in rarray[nrows] */int32_t* rows,
  /* in rarray[nnonzeros] */int32_t* cols,
  /* inout rarray[nnonzeros] */double* values, /* in */int32_t nnonzeros )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
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
  _result = (*(loc_self->d_epv->f_GetValues))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::GetValues( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> ncols,
  /* in rarray[nrows] */::ucxx::sidl::array<int32_t> rows,
  /* in rarray[nnonzeros] */::ucxx::sidl::array<int32_t> cols,
  /* inout rarray[nnonzeros] */::ucxx::sidl::array<double>& values )
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
ucxx::bHYPRE::IJParCSRMatrix::SetRowSizes( /* in rarray[nrows] */int32_t* sizes,
  /* in */int32_t nrows )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  int32_t sizes_lower[1], sizes_upper[1], sizes_stride[1];
  struct sidl_int__array sizes_real;
  struct sidl_int__array *sizes_tmp = &sizes_real;
  sizes_upper[0] = nrows-1;
  sidl_int__array_init(sizes, sizes_tmp, 1, sizes_lower, sizes_upper,
    sizes_stride);
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetRowSizes))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::SetRowSizes( /* in rarray[nrows] 
  */::ucxx::sidl::array<int32_t> sizes )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetRowSizes))(loc_self,
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
ucxx::bHYPRE::IJParCSRMatrix::Print( /* in */const ::std::string& filename )
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
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRMatrix::Read( /* in */const ::std::string& filename,
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
 * Set the int parameter associated with {\tt name}.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRMatrix::SetIntParameter( /* in */const ::std::string& 
  name, /* in */int32_t value )
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
ucxx::bHYPRE::IJParCSRMatrix::SetDoubleParameter( /* in */const ::std::string& 
  name, /* in */double value )
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
ucxx::bHYPRE::IJParCSRMatrix::SetStringParameter( /* in */const ::std::string& 
  name, /* in */const ::std::string& value )
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
ucxx::bHYPRE::IJParCSRMatrix::SetIntArray1Parameter( /* in */const 
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
ucxx::bHYPRE::IJParCSRMatrix::SetIntArray1Parameter( /* in */const 
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
ucxx::bHYPRE::IJParCSRMatrix::SetIntArray2Parameter( /* in */const 
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
ucxx::bHYPRE::IJParCSRMatrix::SetDoubleArray1Parameter( /* in */const 
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
ucxx::bHYPRE::IJParCSRMatrix::SetDoubleArray1Parameter( /* in */const 
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
ucxx::bHYPRE::IJParCSRMatrix::SetDoubleArray2Parameter( /* in */const 
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
ucxx::bHYPRE::IJParCSRMatrix::GetIntValue( /* in */const ::std::string& name,
  /* out */int32_t& value )
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
ucxx::bHYPRE::IJParCSRMatrix::GetDoubleValue( /* in */const ::std::string& name,
  /* out */double& value )
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
ucxx::bHYPRE::IJParCSRMatrix::Setup( /* in */::ucxx::bHYPRE::Vector b,
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
ucxx::bHYPRE::IJParCSRMatrix::Apply( /* in */::ucxx::bHYPRE::Vector b,
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
ucxx::bHYPRE::IJParCSRMatrix::ApplyAdjoint( /* in */::ucxx::bHYPRE::Vector b,
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
  _result = (*(loc_self->d_epv->f_ApplyAdjoint))(loc_self, /* in */ _local_b,
    /* inout */ &_local_x );
  /*dispatch to ior*/
  x._set_ior( _local_x);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */
int32_t
ucxx::bHYPRE::IJParCSRMatrix::GetRow( /* in */int32_t row,
  /* out */int32_t& size, /* out array<int,
  column-major> */::ucxx::sidl::array<int32_t>& col_ind, /* out array<double,
  column-major> */::ucxx::sidl::array<double>& values )
throw ()

{
  int32_t _result;
  ior_t* loc_self = _get_ior();
  struct sidl_int__array* _local_col_ind;
  struct sidl_double__array* _local_values;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetRow))(loc_self, /* in */ row,
    /* out */ &size, /* out array<int,column-major> */ &_local_col_ind,
    /* out array<double,column-major> */ &_local_values );
  /*dispatch to ior*/
  col_ind._set_ior(_local_col_ind);
  values._set_ior(_local_values);
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
::ucxx::bHYPRE::IJParCSRMatrix
ucxx::bHYPRE::IJParCSRMatrix::_create() {
  ::ucxx::bHYPRE::IJParCSRMatrix self( (*_get_ext()->createObject)(), false );
  return self;
}

// copy constructor
ucxx::bHYPRE::IJParCSRMatrix::IJParCSRMatrix ( const 
  ::ucxx::bHYPRE::IJParCSRMatrix& original ) {
  d_self = original._cast("bHYPRE.IJParCSRMatrix");
  d_weak_reference = original.d_weak_reference;
  if (d_self != 0 ) {
    addRef();
  }
}

// assignment operator
::ucxx::bHYPRE::IJParCSRMatrix&
ucxx::bHYPRE::IJParCSRMatrix::operator=( const ::ucxx::bHYPRE::IJParCSRMatrix& 
  rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = rhs._cast("bHYPRE.IJParCSRMatrix");
    d_weak_reference = rhs.d_weak_reference;
    if ( d_self != 0 ) {
      addRef();
    }
  }
  return *this;
}

// conversion from ior to C++ class
ucxx::bHYPRE::IJParCSRMatrix::IJParCSRMatrix ( 
  ::ucxx::bHYPRE::IJParCSRMatrix::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
ucxx::bHYPRE::IJParCSRMatrix::IJParCSRMatrix ( 
  ::ucxx::bHYPRE::IJParCSRMatrix::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// protected method that implements casting
void* ucxx::bHYPRE::IJParCSRMatrix::_cast(const char* type) const
{
  ior_t* loc_self = reinterpret_cast< ior_t*>(this->d_self);
  void* ptr = 0;
  if ( loc_self != 0 ) {
    ptr = reinterpret_cast< void*>((*loc_self->d_epv->f__cast)(loc_self, type));
  }
  return ptr;
}

// Static data type
const ::ucxx::bHYPRE::IJParCSRMatrix::ext_t * 
  ucxx::bHYPRE::IJParCSRMatrix::s_ext = 0;

// private static method to get static data type
const ::ucxx::bHYPRE::IJParCSRMatrix::ext_t *
ucxx::bHYPRE::IJParCSRMatrix::_get_ext()
  throw ( ::ucxx::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = bHYPRE_IJParCSRMatrix__externals();
#else
    s_ext = (struct 
      bHYPRE_IJParCSRMatrix__external*)sidl_dynamicLoadIOR(
      "bHYPRE.IJParCSRMatrix","bHYPRE_IJParCSRMatrix__externals") ;
#endif
  }
  return s_ext;
}

