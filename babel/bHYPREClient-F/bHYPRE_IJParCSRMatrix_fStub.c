/*
 * File:          bHYPRE_IJParCSRMatrix_fStub.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:45 PST
 * Generated:     20030320 16:52:55 PST
 * Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 789
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

#include <stddef.h>
#include <stdlib.h>
#include "SIDLfortran.h"
#include "SIDL_header.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include "SIDL_Loader.h"
#endif
#include "bHYPRE_IJParCSRMatrix_IOR.h"
#include "SIDL_BaseInterface_IOR.h"
#include "SIDL_ClassInfo_IOR.h"
#include "bHYPRE_Vector_IOR.h"

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
    const struct bHYPRE_IJParCSRMatrix__external*(*dll_f)(void) =
      (const struct bHYPRE_IJParCSRMatrix__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "bHYPRE_IJParCSRMatrix__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for bHYPRE.IJParCSRMatrix; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
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
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__cast_f,BHYPRE_IJPARCSRMATRIX__CAST_F,bHYPRE_IJParCSRMatrix__cast_f)
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
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
 * Object in <code>SIDL</code> have an intrinsic reference count.
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
 * Objects in <code>SIDL</code> have an intrinsic reference count.
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
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct SIDL_BaseInterface__object*)
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
 * class.  If the <code>SIDL</code> type name in <code>name</code>
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
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
 * The string name must be the <code>SIDL</code> type name.  This
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
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
  struct SIDL_ClassInfo__object* _proxy_retval = NULL;
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
  int64_t *diag_sizes,
  int64_t *offdiag_sizes,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_diag_sizes = NULL;
  struct SIDL_int__array* _proxy_offdiag_sizes = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_diag_sizes =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*diag_sizes);
  _proxy_offdiag_sizes =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*offdiag_sizes);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDiagOffdSizes))(
      _proxy_self,
      _proxy_diag_sizes,
      _proxy_offdiag_sizes
    );
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
  struct SIDL_int__array* _proxy_col_ind = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
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

/*
 * Set the MPI Communicator.
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
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix_getobject_f,BHYPRE_IJPARCSRMATRIX_GETOBJECT_F,bHYPRE_IJParCSRMatrix_GetObject_f)
(
  int64_t *self,
  int64_t *A,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_A = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetObject))(
      _proxy_self,
      &_proxy_A
    );
  *A = (ptrdiff_t)_proxy_A;
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
 * directly to the column entries in {\tt cols}.  Erases any
 * previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before,
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
  int64_t *ncols,
  int64_t *rows,
  int64_t *cols,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_ncols = NULL;
  struct SIDL_int__array* _proxy_rows = NULL;
  struct SIDL_int__array* _proxy_cols = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_ncols =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*ncols);
  _proxy_rows =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*rows);
  _proxy_cols =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*cols);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetValues))(
      _proxy_self,
      *nrows,
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
  int64_t *ncols,
  int64_t *rows,
  int64_t *cols,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_ncols = NULL;
  struct SIDL_int__array* _proxy_rows = NULL;
  struct SIDL_int__array* _proxy_cols = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_ncols =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*ncols);
  _proxy_rows =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*rows);
  _proxy_cols =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*cols);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddToValues))(
      _proxy_self,
      *nrows,
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
  int64_t *rows,
  int64_t *ncols,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_rows = NULL;
  struct SIDL_int__array* _proxy_ncols = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_rows =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*rows);
  _proxy_ncols =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*ncols);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRowCounts))(
      _proxy_self,
      *nrows,
      _proxy_rows,
      &_proxy_ncols
    );
  *ncols = (ptrdiff_t)_proxy_ncols;
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
  int64_t *ncols,
  int64_t *rows,
  int64_t *cols,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_ncols = NULL;
  struct SIDL_int__array* _proxy_rows = NULL;
  struct SIDL_int__array* _proxy_cols = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_ncols =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*ncols);
  _proxy_rows =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*rows);
  _proxy_cols =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*cols);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetValues))(
      _proxy_self,
      *nrows,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      &_proxy_values
    );
  *values = (ptrdiff_t)_proxy_values;
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  This call can significantly improve the
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
  int64_t *sizes,
  int32_t *retval
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_sizes = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_sizes =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*sizes);
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
    SIDL_copy_fortran_str(SIDL_F77_STR(filename),
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
    SIDL_copy_fortran_str(SIDL_F77_STR(filename),
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
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    SIDL_copy_fortran_str(SIDL_F77_STR(value),
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
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_int__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*value);
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
  struct SIDL_int__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_int__array*)
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
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJParCSRMatrix__epv *_epv = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_double__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*value);
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
  struct SIDL_double__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct SIDL_double__array*)
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
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
    SIDL_interface__array_createCol(*dimen, lower, upper);
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
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_create1d_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATE1D_F,
                  bHYPRE_IJParCSRMatrix__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_create2dcol_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATE2DCOL_F,
                  bHYPRE_IJParCSRMatrix__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_create2drow_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_CREATE2DROW_F,
                  bHYPRE_IJParCSRMatrix__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_addref_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_ADDREF_F,
                  bHYPRE_IJParCSRMatrix__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_deleteref_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_DELETEREF_F,
                  bHYPRE_IJParCSRMatrix__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
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
    SIDL_interface__array_get1((const struct SIDL_interface__array 
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
    SIDL_interface__array_get2((const struct SIDL_interface__array 
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
    SIDL_interface__array_get3((const struct SIDL_interface__array 
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
    SIDL_interface__array_get4((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
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
    SIDL_interface__array_get((const struct SIDL_interface__array 
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
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
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
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
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
  SIDL_interface__array_set3((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
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
  SIDL_interface__array_set4((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_set_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SET_F,
                  bHYPRE_IJParCSRMatrix__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_dimen_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_DIMEN_F,
                  bHYPRE_IJParCSRMatrix__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
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
    SIDL_interface__array_lower((struct SIDL_interface__array 
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
    SIDL_interface__array_upper((struct SIDL_interface__array 
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
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_iscolumnorder_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_IJParCSRMatrix__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_isroworder_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_ISROWORDER_F,
                  bHYPRE_IJParCSRMatrix__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_copy_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_COPY_F,
                  bHYPRE_IJParCSRMatrix__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_ijparcsrmatrix__array_smartcopy_f,
                  BHYPRE_IJPARCSRMATRIX__ARRAY_SMARTCOPY_F,
                  bHYPRE_IJParCSRMatrix__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
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
    SIDL_interface__array_ensure((struct SIDL_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

