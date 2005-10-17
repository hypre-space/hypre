/*
 * File:          bHYPRE_IJParCSRMatrix_fStub.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

/*
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJMatrixView, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

#include <stddef.h>
#include <stdlib.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "bHYPRE_IJParCSRMatrix_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "bHYPRE_Vector_IOR.h"
#include "sidl_BaseInterface_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct bHYPRE_IJParCSRMatrix__external* _getIOR(void)
{
  static const struct bHYPRE_IJParCSRMatrix__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = bHYPRE_IJParCSRMatrix__externals();
#else
    _ior = (struct 
      bHYPRE_IJParCSRMatrix__external*)sidl_dynamicLoadIOR(
      "bHYPRE.IJParCSRMatrix","bHYPRE_IJParCSRMatrix__externals") ;
#endif
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct bHYPRE_IJParCSRMatrix__sepv* _getSEPV(void)
{
  static const struct bHYPRE_IJParCSRMatrix__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__create_f,BHYPRE_IJPARCSRMATRIX__CREATE_F,bHYPRE_IJParCSRMatrix__create_f)
(
  int64_t *self
)
{
  *self = (ptrdiff_t) (*(_getIOR()->createObject))();
}

/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__cast_f,BHYPRE_IJPARCSRMATRIX__CAST_F,bHYPRE_IJParCSRMatrix__cast_f)
(
  int64_t *ref,
  int64_t *retval
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.IJParCSRMatrix");
  } else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__cast2_f,BHYPRE_IJPARCSRMATRIX__CAST2_F,bHYPRE_IJParCSRMatrix__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_addref_f,BHYPRE_IJPARCSRMATRIX_ADDREF_F,bHYPRE_IJParCSRMatrix_addRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_deleteref_f,BHYPRE_IJPARCSRMATRIX_DELETEREF_F,bHYPRE_IJParCSRMatrix_deleteRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_issame_f,BHYPRE_IJPARCSRMATRIX_ISSAME_F,bHYPRE_IJParCSRMatrix_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self,
      _proxy_iobj
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_queryint_f,BHYPRE_IJPARCSRMATRIX_QUERYINT_F,bHYPRE_IJParCSRMatrix_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_istype_f,BHYPRE_IJPARCSRMATRIX_ISTYPE_F,bHYPRE_IJParCSRMatrix_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getclassinfo_f,BHYPRE_IJPARCSRMATRIX_GETCLASSINFO_F,bHYPRE_IJParCSRMatrix_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Method:  Create[]
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_create_f,BHYPRE_IJPARCSRMATRIX_CREATE_F,bHYPRE_IJParCSRMatrix_Create_f)
(
  int64_t *mpi_comm,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *jlower,
  int32_t *jupper,
  int64_t *retval
)
{
  const struct bHYPRE_IJParCSRMatrix__sepv *_epv = _getSEPV();
  void* _proxy_mpi_comm = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_retval = NULL;
  _proxy_mpi_comm =
    (void*)
    (ptrdiff_t)(*mpi_comm);
  _proxy_retval = 
    (*(_epv->f_Create))(
      _proxy_mpi_comm,
      *ilower,
      *iupper,
      *jlower,
      *jupper
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Method:  GenerateLaplacian[]
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_generatelaplacian_f,BHYPRE_IJPARCSRMATRIX_GENERATELAPLACIAN_F,bHYPRE_IJParCSRMatrix_GenerateLaplacian_f)
(
  int64_t *mpi_comm,
  int32_t *nx,
  int32_t *ny,
  int32_t *nz,
  int32_t *Px,
  int32_t *Py,
  int32_t *Pz,
  int32_t *p,
  int32_t *q,
  int32_t *r,
  double *values,
  int32_t *nvalues,
  int32_t *discretization,
  int64_t *retval
)
{
  const struct bHYPRE_IJParCSRMatrix__sepv *_epv = _getSEPV();
  void* _proxy_mpi_comm = NULL;
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct bHYPRE_IJParCSRMatrix__object* _proxy_retval = NULL;
  _proxy_mpi_comm =
    (void*)
    (ptrdiff_t)(*mpi_comm);
  values_upper[0] = (*nvalues)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _proxy_retval = 
    (*(_epv->f_GenerateLaplacian))(
      _proxy_mpi_comm,
      *nx,
      *ny,
      *nz,
      *Px,
      *Py,
      *Pz,
      *p,
      *q,
      *r,
      _proxy_values,
      *discretization
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
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

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setdiagoffdsizes_f,BHYPRE_IJPARCSRMATRIX_SETDIAGOFFDSIZES_F,bHYPRE_IJParCSRMatrix_SetDiagOffdSizes_f)
(
  int64_t *self,
  int32_t *diag_sizes,
  int32_t *offdiag_sizes,
  int32_t *local_nrows,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_diag_sizes;
  struct sidl_int__array* _proxy_diag_sizes = &_alt_diag_sizes;
  int32_t diag_sizes_lower[1], diag_sizes_upper[1], diag_sizes_stride[1];
  struct sidl_int__array _alt_offdiag_sizes;
  struct sidl_int__array* _proxy_offdiag_sizes = &_alt_offdiag_sizes;
  int32_t offdiag_sizes_lower[1], offdiag_sizes_upper[1],
    offdiag_sizes_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  diag_sizes_upper[0] = (*local_nrows)-1;
  sidl_int__array_init(diag_sizes, _proxy_diag_sizes, 1, diag_sizes_lower,
    diag_sizes_upper, diag_sizes_stride);
  offdiag_sizes_upper[0] = (*local_nrows)-1;
  sidl_int__array_init(offdiag_sizes, _proxy_offdiag_sizes, 1,
    offdiag_sizes_lower, offdiag_sizes_upper, offdiag_sizes_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDiagOffdSizes))(
      _proxy_self,
      _proxy_diag_sizes,
      _proxy_offdiag_sizes
    );
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setcommunicator_f,BHYPRE_IJPARCSRMATRIX_SETCOMMUNICATOR_F,bHYPRE_IJParCSRMatrix_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  void* _proxy_mpi_comm = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (void*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self,
      _proxy_mpi_comm
    );
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_initialize_f,BHYPRE_IJPARCSRMATRIX_INITIALIZE_F,bHYPRE_IJParCSRMatrix_Initialize_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Initialize))(
      _proxy_self
    );
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_assemble_f,BHYPRE_IJPARCSRMATRIX_ASSEMBLE_F,bHYPRE_IJParCSRMatrix_Assemble_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Assemble))(
      _proxy_self
    );
}

/*
 * Set the local range for a matrix object.  Each process owns
 * some unique consecutive range of rows, indicated by the
 * global row indices {\tt ilower} and {\tt iupper}.  The row
 * data is required to be such that the value of {\tt ilower} on
 * any process $p$ be exactly one more than the value of {\tt
 * iupper} on process $p-1$.  Note that the first row of the
 * global matrix may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically
 * should match {\tt ilower} and {\tt iupper}, respectively.
 * For rectangular matrices, {\tt jlower} and {\tt jupper}
 * should define a partitioning of the columns.  This
 * partitioning must be used for any vector $v$ that will be
 * used in matrix-vector products with the rectangular matrix.
 * The matrix data structure may use {\tt jlower} and {\tt
 * jupper} to store the diagonal blocks (rectangular in general)
 * of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setlocalrange_f,BHYPRE_IJPARCSRMATRIX_SETLOCALRANGE_F,bHYPRE_IJParCSRMatrix_SetLocalRange_f)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLocalRange))(
      _proxy_self,
      *ilower,
      *iupper,
      *jlower,
      *jupper
    );
}

/*
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

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setvalues_f,BHYPRE_IJPARCSRMATRIX_SETVALUES_F,bHYPRE_IJParCSRMatrix_SetValues_f)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *ncols,
  int32_t *rows,
  int32_t *cols,
  double *values,
  int32_t *nnonzeros,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_ncols;
  struct sidl_int__array* _proxy_ncols = &_alt_ncols;
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array _alt_rows;
  struct sidl_int__array* _proxy_rows = &_alt_rows;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array _alt_cols;
  struct sidl_int__array* _proxy_cols = &_alt_cols;
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  ncols_upper[0] = (*nrows)-1;
  sidl_int__array_init(ncols, _proxy_ncols, 1, ncols_lower, ncols_upper,
    ncols_stride);
  rows_upper[0] = (*nrows)-1;
  sidl_int__array_init(rows, _proxy_rows, 1, rows_lower, rows_upper,
    rows_stride);
  cols_upper[0] = (*nnonzeros)-1;
  sidl_int__array_init(cols, _proxy_cols, 1, cols_lower, cols_upper,
    cols_stride);
  values_upper[0] = (*nnonzeros)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetValues))(
      _proxy_self,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values
    );
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_addtovalues_f,BHYPRE_IJPARCSRMATRIX_ADDTOVALUES_F,bHYPRE_IJParCSRMatrix_AddToValues_f)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *ncols,
  int32_t *rows,
  int32_t *cols,
  double *values,
  int32_t *nnonzeros,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_ncols;
  struct sidl_int__array* _proxy_ncols = &_alt_ncols;
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array _alt_rows;
  struct sidl_int__array* _proxy_rows = &_alt_rows;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array _alt_cols;
  struct sidl_int__array* _proxy_cols = &_alt_cols;
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  ncols_upper[0] = (*nrows)-1;
  sidl_int__array_init(ncols, _proxy_ncols, 1, ncols_lower, ncols_upper,
    ncols_stride);
  rows_upper[0] = (*nrows)-1;
  sidl_int__array_init(rows, _proxy_rows, 1, rows_lower, rows_upper,
    rows_stride);
  cols_upper[0] = (*nnonzeros)-1;
  sidl_int__array_init(cols, _proxy_cols, 1, cols_lower, cols_upper,
    cols_stride);
  values_upper[0] = (*nnonzeros)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddToValues))(
      _proxy_self,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values
    );
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getlocalrange_f,BHYPRE_IJPARCSRMATRIX_GETLOCALRANGE_F,bHYPRE_IJParCSRMatrix_GetLocalRange_f)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetLocalRange))(
      _proxy_self,
      ilower,
      iupper,
      jlower,
      jupper
    );
}

/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getrowcounts_f,BHYPRE_IJPARCSRMATRIX_GETROWCOUNTS_F,bHYPRE_IJParCSRMatrix_GetRowCounts_f)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *rows,
  int32_t *ncols,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_rows;
  struct sidl_int__array* _proxy_rows = &_alt_rows;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array _alt_ncols;
  struct sidl_int__array* _proxy_ncols = &_alt_ncols;
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  rows_upper[0] = (*nrows)-1;
  sidl_int__array_init(rows, _proxy_rows, 1, rows_lower, rows_upper,
    rows_stride);
  ncols_upper[0] = (*nrows)-1;
  sidl_int__array_init(ncols, _proxy_ncols, 1, ncols_lower, ncols_upper,
    ncols_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRowCounts))(
      _proxy_self,
      _proxy_rows,
      &_proxy_ncols
    );
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getvalues_f,BHYPRE_IJPARCSRMATRIX_GETVALUES_F,bHYPRE_IJParCSRMatrix_GetValues_f)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *ncols,
  int32_t *rows,
  int32_t *cols,
  double *values,
  int32_t *nnonzeros,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_ncols;
  struct sidl_int__array* _proxy_ncols = &_alt_ncols;
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array _alt_rows;
  struct sidl_int__array* _proxy_rows = &_alt_rows;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array _alt_cols;
  struct sidl_int__array* _proxy_cols = &_alt_cols;
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  ncols_upper[0] = (*nrows)-1;
  sidl_int__array_init(ncols, _proxy_ncols, 1, ncols_lower, ncols_upper,
    ncols_stride);
  rows_upper[0] = (*nrows)-1;
  sidl_int__array_init(rows, _proxy_rows, 1, rows_lower, rows_upper,
    rows_stride);
  cols_upper[0] = (*nnonzeros)-1;
  sidl_int__array_init(cols, _proxy_cols, 1, cols_lower, cols_upper,
    cols_stride);
  values_upper[0] = (*nnonzeros)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetValues))(
      _proxy_self,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      &_proxy_values
    );
}

/*
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

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setrowsizes_f,BHYPRE_IJPARCSRMATRIX_SETROWSIZES_F,bHYPRE_IJParCSRMatrix_SetRowSizes_f)
(
  int64_t *self,
  int32_t *sizes,
  int32_t *nrows,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_int__array _alt_sizes;
  struct sidl_int__array* _proxy_sizes = &_alt_sizes;
  int32_t sizes_lower[1], sizes_upper[1], sizes_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  sizes_upper[0] = (*nrows)-1;
  sidl_int__array_init(sizes, _proxy_sizes, 1, sizes_lower, sizes_upper,
    sizes_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetRowSizes))(
      _proxy_self,
      _proxy_sizes
    );
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_print_f,BHYPRE_IJPARCSRMATRIX_PRINT_F,bHYPRE_IJParCSRMatrix_Print_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    sidl_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Print))(
      _proxy_self,
      _proxy_filename
    );
  free((void *)_proxy_filename);
}

/*
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_read_f,BHYPRE_IJPARCSRMATRIX_READ_F,bHYPRE_IJParCSRMatrix_Read_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int64_t *comm,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  void* _proxy_comm = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    sidl_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _proxy_comm =
    (void*)
    (ptrdiff_t)(*comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Read))(
      _proxy_self,
      _proxy_filename,
      _proxy_comm
    );
  free((void *)_proxy_filename);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setintparameter_f,BHYPRE_IJPARCSRMATRIX_SETINTPARAMETER_F,bHYPRE_IJParCSRMatrix_SetIntParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setdoubleparameter_f,BHYPRE_IJPARCSRMATRIX_SETDOUBLEPARAMETER_F,bHYPRE_IJParCSRMatrix_SetDoubleParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setstringparameter_f,BHYPRE_IJPARCSRMATRIX_SETSTRINGPARAMETER_F,bHYPRE_IJParCSRMatrix_SetStringParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_String value
  SIDL_F77_STR_NEAR_LEN_DECL(value),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
  SIDL_F77_STR_FAR_LEN_DECL(value)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  char* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    sidl_copy_fortran_str(SIDL_F77_STR(value),
      SIDL_F77_STR_LEN(value));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetStringParameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
  free((void *)_proxy_value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setintarray1parameter_f,BHYPRE_IJPARCSRMATRIX_SETINTARRAY1PARAMETER_F,bHYPRE_IJParCSRMatrix_SetIntArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *nvalues,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array _alt_value;
  struct sidl_int__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  value_upper[0] = (*nvalues)-1;
  sidl_int__array_init(value, _proxy_value, 1, value_lower, value_upper,
    value_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setintarray2parameter_f,BHYPRE_IJPARCSRMATRIX_SETINTARRAY2PARAMETER_F,bHYPRE_IJParCSRMatrix_SetIntArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct sidl_int__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setdoublearray1parameter_f,BHYPRE_IJPARCSRMATRIX_SETDOUBLEARRAY1PARAMETER_F,bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *nvalues,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array _alt_value;
  struct sidl_double__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  value_upper[0] = (*nvalues)-1;
  sidl_double__array_init(value, _proxy_value, 1, value_lower, value_upper,
    value_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setdoublearray2parameter_f,BHYPRE_IJPARCSRMATRIX_SETDOUBLEARRAY2PARAMETER_F,bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct sidl_double__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getintvalue_f,BHYPRE_IJPARCSRMATRIX_GETINTVALUE_F,bHYPRE_IJParCSRMatrix_GetIntValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetIntValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getdoublevalue_f,BHYPRE_IJPARCSRMATRIX_GETDOUBLEVALUE_F,bHYPRE_IJParCSRMatrix_GetDoubleValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetDoubleValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_setup_f,BHYPRE_IJPARCSRMATRIX_SETUP_F,bHYPRE_IJParCSRMatrix_Setup_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Setup))(
      _proxy_self,
      _proxy_b,
      _proxy_x
    );
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_apply_f,BHYPRE_IJPARCSRMATRIX_APPLY_F,bHYPRE_IJParCSRMatrix_Apply_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Apply))(
      _proxy_self,
      _proxy_b,
      &_proxy_x
    );
  *x = (ptrdiff_t)_proxy_x;
}

/*
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getrow_f,BHYPRE_IJPARCSRMATRIX_GETROW_F,bHYPRE_IJParCSRMatrix_GetRow_f)
(
  int64_t *self,
  int32_t *row,
  int32_t *size,
  int64_t *col_ind,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_col_ind = NULL;
  struct sidl_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRow))(
      _proxy_self,
      *row,
      size,
      &_proxy_col_ind,
      &_proxy_values
    );
  *col_ind = (ptrdiff_t)_proxy_col_ind;
  *values = (ptrdiff_t)_proxy_values;
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_createcol_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATECOL_F,
                  bHYPRE_IJParCSRMatrix__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_createrow_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATEROW_F,
                  bHYPRE_IJParCSRMatrix__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_create1d_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATE1D_F,
                  bHYPRE_IJParCSRMatrix__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_create2dcol_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATE2DCOL_F,
                  bHYPRE_IJParCSRMatrix__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_create2drow_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATE2DROW_F,
                  bHYPRE_IJParCSRMatrix__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_addref_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_ADDREF_F,
                  bHYPRE_IJParCSRMatrix__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_deleteref_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_DELETEREF_F,
                  bHYPRE_IJParCSRMatrix__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get1_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET1_F,
                  bHYPRE_IJParCSRMatrix__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get1((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get2_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET2_F,
                  bHYPRE_IJParCSRMatrix__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get2((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get3_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET3_F,
                  bHYPRE_IJParCSRMatrix__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get3((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get4_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET4_F,
                  bHYPRE_IJParCSRMatrix__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get4((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get5_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET5_F,
                  bHYPRE_IJParCSRMatrix__array_get5_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get5((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get6_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET6_F,
                  bHYPRE_IJParCSRMatrix__array_get6_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get6((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get7_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET7_F,
                  bHYPRE_IJParCSRMatrix__array_get7_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get7((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_get_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_GET_F,
                  bHYPRE_IJParCSRMatrix__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set1_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET1_F,
                  bHYPRE_IJParCSRMatrix__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set2_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET2_F,
                  bHYPRE_IJParCSRMatrix__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set3_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET3_F,
                  bHYPRE_IJParCSRMatrix__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set4_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET4_F,
                  bHYPRE_IJParCSRMatrix__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set5_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET5_F,
                  bHYPRE_IJParCSRMatrix__array_set5_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int64_t *value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set6_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET6_F,
                  bHYPRE_IJParCSRMatrix__array_set6_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int64_t *value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set7_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET7_F,
                  bHYPRE_IJParCSRMatrix__array_set7_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int64_t *value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET_F,
                  bHYPRE_IJParCSRMatrix__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_dimen_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_DIMEN_F,
                  bHYPRE_IJParCSRMatrix__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_lower_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_LOWER_F,
                  bHYPRE_IJParCSRMatrix__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_upper_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_UPPER_F,
                  bHYPRE_IJParCSRMatrix__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_length_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_LENGTH_F,
                  bHYPRE_IJParCSRMatrix__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_stride_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_STRIDE_F,
                  bHYPRE_IJParCSRMatrix__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_iscolumnorder_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_IJParCSRMatrix__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_isroworder_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_ISROWORDER_F,
                  bHYPRE_IJParCSRMatrix__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_copy_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_COPY_F,
                  bHYPRE_IJParCSRMatrix__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_smartcopy_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SMARTCOPY_F,
                  bHYPRE_IJParCSRMatrix__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_slice_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SLICE_F,
                  bHYPRE_IJParCSRMatrix__array_slice_f)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_slice((struct sidl_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_ensure_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_ENSURE_F,
                  bHYPRE_IJParCSRMatrix__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_ensure((struct sidl_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

