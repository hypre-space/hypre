/*
 * File:          Hypre_IJBuildMatrix_fStub.c
 * Symbol:        Hypre.IJBuildMatrix-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:22 PST
 * Generated:     20030121 14:39:30 PST
 * Description:   Client-side glue code for Hypre.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 155
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * Symbol "Hypre.IJBuildMatrix" (version 0.1.6)
 * 
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 * 
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
#include "Hypre_IJBuildMatrix_IOR.h"
#include "SIDL_BaseInterface_IOR.h"

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__cast_f,HYPRE_IJBUILDMATRIX__CAST_F,Hypre_IJBuildMatrix__cast_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(hypre_ijbuildmatrix_addref_f,HYPRE_IJBUILDMATRIX_ADDREF_F,Hypre_IJBuildMatrix_addRef_f)
(
  int64_t *self
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self->d_object
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
SIDLFortran77Symbol(hypre_ijbuildmatrix_deleteref_f,HYPRE_IJBUILDMATRIX_DELETEREF_F,Hypre_IJBuildMatrix_deleteRef_f)
(
  int64_t *self
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self->d_object
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_issame_f,HYPRE_IJBUILDMATRIX_ISSAME_F,Hypre_IJBuildMatrix_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct SIDL_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(hypre_ijbuildmatrix_queryint_f,HYPRE_IJBUILDMATRIX_QUERYINT_F,Hypre_IJBuildMatrix_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(hypre_ijbuildmatrix_istype_f,HYPRE_IJBUILDMATRIX_ISTYPE_F,Hypre_IJBuildMatrix_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self->d_object,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Method:  SetCommunicator[]
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_setcommunicator_f,HYPRE_IJBUILDMATRIX_SETCOMMUNICATOR_F,Hypre_IJBuildMatrix_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  void* _proxy_mpi_comm = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (void*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self->d_object,
      _proxy_mpi_comm
    );
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_initialize_f,HYPRE_IJBUILDMATRIX_INITIALIZE_F,Hypre_IJBuildMatrix_Initialize_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Initialize))(
      _proxy_self->d_object
    );
}

/*
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_assemble_f,HYPRE_IJBUILDMATRIX_ASSEMBLE_F,Hypre_IJBuildMatrix_Assemble_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Assemble))(
      _proxy_self->d_object
    );
}

/*
 * The problem definition interface is a "builder" that creates an object
 * that contains the problem definition information, e.g. a matrix. To
 * perform subsequent operations with that object, it must be returned from
 * the problem definition object. "GetObject" performs this function.
 * <note>At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface. QueryInterface or Cast must
 * be used on the returned object to convert it into a known type.</note>
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_getobject_f,HYPRE_IJBUILDMATRIX_GETOBJECT_F,Hypre_IJBuildMatrix_GetObject_f)
(
  int64_t *self,
  int64_t *A,
  int32_t *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_A = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetObject))(
      _proxy_self->d_object,
      &_proxy_A
    );
  *A = (ptrdiff_t)_proxy_A;
}

/*
 * Create a matrix object.  Each process owns some unique consecutive
 * range of rows, indicated by the global row indices {\tt ilower} and
 * {\tt iupper}.  The row data is required to be such that the value
 * of {\tt ilower} on any process $p$ be exactly one more than the
 * value of {\tt iupper} on process $p-1$.  Note that the first row of
 * the global matrix may start with any integer value.  In particular,
 * one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically should
 * match {\tt ilower} and {\tt iupper}, respectively.  For rectangular
 * matrices, {\tt jlower} and {\tt jupper} should define a
 * partitioning of the columns.  This partitioning must be used for
 * any vector $v$ that will be used in matrix-vector products with the
 * rectangular matrix.  The matrix data structure may use {\tt jlower}
 * and {\tt jupper} to store the diagonal blocks (rectangular in
 * general) of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_create_f,HYPRE_IJBUILDMATRIX_CREATE_F,Hypre_IJBuildMatrix_Create_f)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Create))(
      _proxy_self->d_object,
      *ilower,
      *iupper,
      *jlower,
      *jupper
    );
}

/*
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt ncols}
 * and {\tt rows} are of dimension {\tt nrows} and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array {\tt cols} contains the column indices for each of the {\tt
 * rows}, and is ordered by rows.  The data in the {\tt values} array
 * corresponds directly to the column entries in {\tt cols}.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one.
 * 
 * Not collective.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_setvalues_f,HYPRE_IJBUILDMATRIX_SETVALUES_F,Hypre_IJBuildMatrix_SetValues_f)
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
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_ncols = NULL;
  struct SIDL_int__array* _proxy_rows = NULL;
  struct SIDL_int__array* _proxy_cols = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
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
      _proxy_self->d_object,
      *nrows,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values
    );
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value there
 * before, inserts a new one.
 * 
 * Not collective.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_addtovalues_f,HYPRE_IJBUILDMATRIX_ADDTOVALUES_F,Hypre_IJBuildMatrix_AddToValues_f)
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
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_ncols = NULL;
  struct SIDL_int__array* _proxy_rows = NULL;
  struct SIDL_int__array* _proxy_cols = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
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
      _proxy_self->d_object,
      *nrows,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values
    );
}

/*
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array {\tt sizes} contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * DEVELOPER NOTES: None.
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_setrowsizes_f,HYPRE_IJBUILDMATRIX_SETROWSIZES_F,Hypre_IJBuildMatrix_SetRowSizes_f)
(
  int64_t *self,
  int64_t *sizes,
  int32_t *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_sizes = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_sizes =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*sizes);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetRowSizes))(
      _proxy_self->d_object,
      _proxy_sizes
    );
}

/*
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * {\tt diag\_sizes} and {\tt offdiag\_sizes} contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_setdiagoffdsizes_f,HYPRE_IJBUILDMATRIX_SETDIAGOFFDSIZES_F,Hypre_IJBuildMatrix_SetDiagOffdSizes_f)
(
  int64_t *self,
  int64_t *diag_sizes,
  int64_t *offdiag_sizes,
  int32_t *retval
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_diag_sizes = NULL;
  struct SIDL_int__array* _proxy_offdiag_sizes = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
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
      _proxy_self->d_object,
      _proxy_diag_sizes,
      _proxy_offdiag_sizes
    );
}

/*
 * Read the matrix from file.  This is mainly for debugging purposes.
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_read_f,HYPRE_IJBUILDMATRIX_READ_F,Hypre_IJBuildMatrix_Read_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int64_t *comm,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  void* _proxy_comm = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
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
      _proxy_self->d_object,
      _proxy_filename,
      _proxy_comm
    );
  free((void *)_proxy_filename);
}

/*
 * Print the matrix to file.  This is mainly for debugging purposes.
 * 
 */

void
SIDLFortran77Symbol(hypre_ijbuildmatrix_print_f,HYPRE_IJBUILDMATRIX_PRINT_F,Hypre_IJBuildMatrix_Print_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct Hypre_IJBuildMatrix__epv *_epv = NULL;
  struct Hypre_IJBuildMatrix__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  _proxy_self =
    (struct Hypre_IJBuildMatrix__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    SIDL_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Print))(
      _proxy_self->d_object,
      _proxy_filename
    );
  free((void *)_proxy_filename);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_createcol_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_CREATECOL_F,
                  Hypre_IJBuildMatrix__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_createrow_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_CREATEROW_F,
                  Hypre_IJBuildMatrix__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_create1d_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_CREATE1D_F,
                  Hypre_IJBuildMatrix__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_create2dcol_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_CREATE2DCOL_F,
                  Hypre_IJBuildMatrix__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_create2drow_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_CREATE2DROW_F,
                  Hypre_IJBuildMatrix__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_addref_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_ADDREF_F,
                  Hypre_IJBuildMatrix__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_deleteref_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_DELETEREF_F,
                  Hypre_IJBuildMatrix__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_get1_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_GET1_F,
                  Hypre_IJBuildMatrix__array_get1_f)
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
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_get2_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_GET2_F,
                  Hypre_IJBuildMatrix__array_get2_f)
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
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_get3_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_GET3_F,
                  Hypre_IJBuildMatrix__array_get3_f)
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
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_get4_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_GET4_F,
                  Hypre_IJBuildMatrix__array_get4_f)
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
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_get_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_GET_F,
                  Hypre_IJBuildMatrix__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_set1_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_SET1_F,
                  Hypre_IJBuildMatrix__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_set2_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_SET2_F,
                  Hypre_IJBuildMatrix__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_set3_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_SET3_F,
                  Hypre_IJBuildMatrix__array_set3_f)
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
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_set4_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_SET4_F,
                  Hypre_IJBuildMatrix__array_set4_f)
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
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_set_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_SET_F,
                  Hypre_IJBuildMatrix__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_dimen_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_DIMEN_F,
                  Hypre_IJBuildMatrix__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_lower_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_LOWER_F,
                  Hypre_IJBuildMatrix__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_lower((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_upper_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_UPPER_F,
                  Hypre_IJBuildMatrix__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_upper((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_stride_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_STRIDE_F,
                  Hypre_IJBuildMatrix__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_iscolumnorder_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_ISCOLUMNORDER_F,
                  Hypre_IJBuildMatrix__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_isroworder_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_ISROWORDER_F,
                  Hypre_IJBuildMatrix__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_copy_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_COPY_F,
                  Hypre_IJBuildMatrix__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_smartcopy_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_SMARTCOPY_F,
                  Hypre_IJBuildMatrix__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(hypre_ijbuildmatrix__array_ensure_f,
                  HYPRE_IJBUILDMATRIX__ARRAY_ENSURE_F,
                  Hypre_IJBuildMatrix__array_ensure_f)
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

