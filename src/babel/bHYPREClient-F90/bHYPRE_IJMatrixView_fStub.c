/*
 * File:          bHYPRE_IJMatrixView_fStub.c
 * Symbol:        bHYPRE.IJMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.IJMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

/*
 * Symbol "bHYPRE.IJMatrixView" (version 1.0.0)
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 */

#ifndef included_bHYPRE_IJMatrixView_fStub_h
#include "bHYPRE_IJMatrixView_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
#ifndef included_sidlf90array_h
#include "sidlf90array.h"
#endif
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_Exception_h
#include "sidl_Exception.h"
#endif
#include <stdio.h>
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "bHYPRE_IJMatrixView_IOR.h"
#include "bHYPRE_IJMatrixView_fAbbrev.h"
#include "bHYPRE_MPICommunicator_IOR.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_bHYPRE_IJMatrixView_fStub_h
#include "bHYPRE_IJMatrixView_fStub.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_fStub_h
#include "bHYPRE_MPICommunicator_fStub.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_fStub_h
#include "bHYPRE_MatrixVectorView_fStub.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_fStub_h
#include "bHYPRE_ProblemDefinition_fStub.h"
#endif
#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct bHYPRE_IJMatrixView__object* 
  bHYPRE_IJMatrixView__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct bHYPRE_IJMatrixView__object* 
  bHYPRE_IJMatrixView__IHConnect(struct sidl_rmi_InstanceHandle__object 
  *instance, struct sidl_BaseInterface__object **_ex);
/*
 * Remote Connector for the class.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_rconnect_m,BHYPRE_IJMATRIXVIEW_RCONNECT_M,bHYPRE_IJMatrixView_rConnect_m)
(
  int64_t *self,
  SIDL_F90_String url
  SIDL_F90_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(url)
)
{
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F90_STR(url),
      SIDL_F90_STR_LEN(url));
  _proxy_self = bHYPRE_IJMatrixView__remoteConnect(_proxy_url, 1,
    &_proxy_exception);
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *self = (ptrdiff_t)_proxy_self;
  }
  free((void *)_proxy_url);
}
/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview__cast_m,BHYPRE_IJMATRIXVIEW__CAST_M,bHYPRE_IJMatrixView__cast_m)
(
  int64_t *ref,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  struct sidl_BaseInterface__object *proxy_exception;

  *retval = 0;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.IJMatrixView",
      (void*)bHYPRE_IJMatrixView__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.IJMatrixView", &proxy_exception);
  } else {
    *retval = 0;
    proxy_exception = 0;
  }
  EXIT:
  *exception = (ptrdiff_t)proxy_exception;
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview__cast2_m,BHYPRE_IJMATRIXVIEW__CAST2_M,bHYPRE_IJMatrixView__cast2_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self->d_object,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_name);
}


/*
 * Select and execute a method by name
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview__exec_m,BHYPRE_IJMATRIXVIEW__EXEC_M,bHYPRE_IJMatrixView__exec_m)
(
  int64_t *self,
  SIDL_F90_String methodName
  SIDL_F90_STR_NEAR_LEN_DECL(methodName),
  int64_t *inArgs,
  int64_t *outArgs,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(methodName)
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_methodName =
    sidl_copy_fortran_str(SIDL_F90_STR(methodName),
      SIDL_F90_STR_LEN(methodName));
  _proxy_inArgs =
    (struct sidl_rmi_Call__object*)
    (ptrdiff_t)(*inArgs);
  _proxy_outArgs =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*outArgs);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__exec))(
    _proxy_self->d_object,
    _proxy_methodName,
    _proxy_inArgs,
    _proxy_outArgs,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_methodName);
}


/*
 * Get the URL of the Implementation of this object (for RMI)
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview__geturl_m,BHYPRE_IJMATRIXVIEW__GETURL_M,bHYPRE_IJMatrixView__getURL_m)
(
  int64_t *self,
  SIDL_F90_String retval
  SIDL_F90_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(retval)
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__getURL))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    sidl_copy_c_str(
      SIDL_F90_STR(retval),
      SIDL_F90_STR_LEN(retval),
      _proxy_retval);
  }
  free((void *)_proxy_retval);
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview__isremote_m,BHYPRE_IJMATRIXVIEW__ISREMOTE_M,bHYPRE_IJMatrixView__isRemote_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__isRemote))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview__islocal_m,BHYPRE_IJMATRIXVIEW__ISLOCAL_M,bHYPRE_IJMatrixView__isLocal_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    !(*(_epv->f__isRemote))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview__set_hooks_m,BHYPRE_IJMATRIXVIEW__SET_HOOKS_M,bHYPRE_IJMatrixView__set_hooks_m)
(
  int64_t *self,
  SIDL_F90_Bool *on,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_on = ((*on == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__set_hooks))(
    _proxy_self->d_object,
    _proxy_on,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
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
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_setlocalrange_m,BHYPRE_IJMATRIXVIEW_SETLOCALRANGE_M,bHYPRE_IJMatrixView_SetLocalRange_m)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLocalRange))(
      _proxy_self->d_object,
      *ilower,
      *iupper,
      *jlower,
      *jupper,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
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
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_setvalues_m,BHYPRE_IJMATRIXVIEW_SETVALUES_M,bHYPRE_IJMatrixView_SetValues_m)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *ncols,
  int32_t *rows,
  int32_t *cols,
  double *values,
  int32_t *nnonzeros,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
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
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
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
      _proxy_self->d_object,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
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
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_setvalues_a,BHYPRE_IJMATRIXVIEW_SETVALUES_A,bHYPRE_IJMatrixView_SetValues_a)
(
  int64_t *self,
  struct sidl_fortran_array *ncols,
  struct sidl_fortran_array *rows,
  struct sidl_fortran_array *cols,
  struct sidl_fortran_array *values,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_ncols = NULL;
  struct sidl_int__array* _proxy_rows = NULL;
  struct sidl_int__array* _proxy_cols = NULL;
  struct sidl_double__array* _proxy_values = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_ncols =
    (struct sidl_int__array*)
    (ptrdiff_t)(ncols->d_ior);
  _proxy_rows =
    (struct sidl_int__array*)
    (ptrdiff_t)(rows->d_ior);
  _proxy_cols =
    (struct sidl_int__array*)
    (ptrdiff_t)(cols->d_ior);
  _proxy_values =
    (struct sidl_double__array*)
    (ptrdiff_t)(values->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetValues))(
      _proxy_self->d_object,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_addtovalues_m,BHYPRE_IJMATRIXVIEW_ADDTOVALUES_M,bHYPRE_IJMatrixView_AddToValues_m)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *ncols,
  int32_t *rows,
  int32_t *cols,
  double *values,
  int32_t *nnonzeros,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
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
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
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
      _proxy_self->d_object,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}
/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_addtovalues_a,BHYPRE_IJMATRIXVIEW_ADDTOVALUES_A,bHYPRE_IJMatrixView_AddToValues_a)
(
  int64_t *self,
  struct sidl_fortran_array *ncols,
  struct sidl_fortran_array *rows,
  struct sidl_fortran_array *cols,
  struct sidl_fortran_array *values,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_ncols = NULL;
  struct sidl_int__array* _proxy_rows = NULL;
  struct sidl_int__array* _proxy_cols = NULL;
  struct sidl_double__array* _proxy_values = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_ncols =
    (struct sidl_int__array*)
    (ptrdiff_t)(ncols->d_ior);
  _proxy_rows =
    (struct sidl_int__array*)
    (ptrdiff_t)(rows->d_ior);
  _proxy_cols =
    (struct sidl_int__array*)
    (ptrdiff_t)(cols->d_ior);
  _proxy_values =
    (struct sidl_double__array*)
    (ptrdiff_t)(values->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddToValues))(
      _proxy_self->d_object,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      _proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_getlocalrange_m,BHYPRE_IJMATRIXVIEW_GETLOCALRANGE_M,bHYPRE_IJMatrixView_GetLocalRange_m)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetLocalRange))(
      _proxy_self->d_object,
      ilower,
      iupper,
      jlower,
      jupper,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_getrowcounts_m,BHYPRE_IJMATRIXVIEW_GETROWCOUNTS_M,bHYPRE_IJMatrixView_GetRowCounts_m)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *rows,
  int32_t *ncols,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_int__array _alt_rows;
  struct sidl_int__array* _proxy_rows = &_alt_rows;
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array _alt_ncols;
  struct sidl_int__array* _proxy_ncols = &_alt_ncols;
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
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
      _proxy_self->d_object,
      _proxy_rows,
      &_proxy_ncols,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}
/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_getrowcounts_a,BHYPRE_IJMATRIXVIEW_GETROWCOUNTS_A,bHYPRE_IJMatrixView_GetRowCounts_a)
(
  int64_t *self,
  struct sidl_fortran_array *rows,
  struct sidl_fortran_array *ncols,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_rows = NULL;
  struct sidl_int__array* _proxy_ncols = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_rows =
    (struct sidl_int__array*)
    (ptrdiff_t)(rows->d_ior);
  _proxy_ncols =
    (struct sidl_int__array*)
    (ptrdiff_t)(ncols->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRowCounts))(
      _proxy_self->d_object,
      _proxy_rows,
      &_proxy_ncols,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_int__array_convert2f90(_proxy_ncols, 1, ncols)) {
      /* Copy to contiguous column-order */
      struct sidl_int__array* _alt_ncols =
        sidl_int__array_ensure(_proxy_ncols, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_ncols);
      if (sidl_int__array_convert2f90(_alt_ncols, 1, ncols)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_ncols, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_getvalues_m,BHYPRE_IJMATRIXVIEW_GETVALUES_M,bHYPRE_IJMatrixView_GetValues_m)
(
  int64_t *self,
  int32_t *nrows,
  int32_t *ncols,
  int32_t *rows,
  int32_t *cols,
  double *values,
  int32_t *nnonzeros,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
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
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
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
      _proxy_self->d_object,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      &_proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}
/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_getvalues_a,BHYPRE_IJMATRIXVIEW_GETVALUES_A,bHYPRE_IJMatrixView_GetValues_a)
(
  int64_t *self,
  struct sidl_fortran_array *ncols,
  struct sidl_fortran_array *rows,
  struct sidl_fortran_array *cols,
  struct sidl_fortran_array *values,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_ncols = NULL;
  struct sidl_int__array* _proxy_rows = NULL;
  struct sidl_int__array* _proxy_cols = NULL;
  struct sidl_double__array* _proxy_values = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_ncols =
    (struct sidl_int__array*)
    (ptrdiff_t)(ncols->d_ior);
  _proxy_rows =
    (struct sidl_int__array*)
    (ptrdiff_t)(rows->d_ior);
  _proxy_cols =
    (struct sidl_int__array*)
    (ptrdiff_t)(cols->d_ior);
  _proxy_values =
    (struct sidl_double__array*)
    (ptrdiff_t)(values->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetValues))(
      _proxy_self->d_object,
      _proxy_ncols,
      _proxy_rows,
      _proxy_cols,
      &_proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    if (sidl_double__array_convert2f90(_proxy_values, 1, values)) {
      /* Copy to contiguous column-order */
      struct sidl_double__array* _alt_values =
        sidl_double__array_ensure(_proxy_values, 1,
          sidl_column_major_order);
      sidl__array_deleteRef((struct sidl__array *)_proxy_values);
      if (sidl_double__array_convert2f90(_alt_values, 1, values)) {
        /* We're S.O.L. */
        fprintf(stderr, "convert2f90 failed: %p %d\n", (void*)_alt_values, 1);
        exit(1); /*NOTREACHED*/
      }
    }
  }
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
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_setrowsizes_m,BHYPRE_IJMATRIXVIEW_SETROWSIZES_M,bHYPRE_IJMatrixView_SetRowSizes_m)
(
  int64_t *self,
  int32_t *sizes,
  int32_t *nrows,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_int__array _alt_sizes;
  struct sidl_int__array* _proxy_sizes = &_alt_sizes;
  int32_t sizes_lower[1], sizes_upper[1], sizes_stride[1];
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  sizes_upper[0] = (*nrows)-1;
  sidl_int__array_init(sizes, _proxy_sizes, 1, sizes_lower, sizes_upper,
    sizes_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetRowSizes))(
      _proxy_self->d_object,
      _proxy_sizes,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
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
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_setrowsizes_a,BHYPRE_IJMATRIXVIEW_SETROWSIZES_A,bHYPRE_IJMatrixView_SetRowSizes_a)
(
  int64_t *self,
  struct sidl_fortran_array *sizes,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_sizes = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_sizes =
    (struct sidl_int__array*)
    (ptrdiff_t)(sizes->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetRowSizes))(
      _proxy_self->d_object,
      _proxy_sizes,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_print_m,BHYPRE_IJMATRIXVIEW_PRINT_M,bHYPRE_IJMatrixView_Print_m)
(
  int64_t *self,
  SIDL_F90_String filename
  SIDL_F90_STR_NEAR_LEN_DECL(filename),
  int32_t *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(filename)
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    sidl_copy_fortran_str(SIDL_F90_STR(filename),
      SIDL_F90_STR_LEN(filename));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Print))(
      _proxy_self->d_object,
      _proxy_filename,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_filename);
}

/*
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_read_m,BHYPRE_IJMATRIXVIEW_READ_M,bHYPRE_IJMatrixView_Read_m)
(
  int64_t *self,
  SIDL_F90_String filename
  SIDL_F90_STR_NEAR_LEN_DECL(filename),
  int64_t *comm,
  int32_t *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(filename)
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  struct bHYPRE_MPICommunicator__object* _proxy_comm = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    sidl_copy_fortran_str(SIDL_F90_STR(filename),
      SIDL_F90_STR_LEN(filename));
  _proxy_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Read))(
      _proxy_self->d_object,
      _proxy_filename,
      _proxy_comm,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_filename);
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_setcommunicator_m,BHYPRE_IJMATRIXVIEW_SETCOMMUNICATOR_M,bHYPRE_IJMatrixView_SetCommunicator_m)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self->d_object,
      _proxy_mpi_comm,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_destroy_m,BHYPRE_IJMATRIXVIEW_DESTROY_M,bHYPRE_IJMatrixView_Destroy_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_Destroy))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_initialize_m,BHYPRE_IJMATRIXVIEW_INITIALIZE_M,bHYPRE_IJMatrixView_Initialize_m)
(
  int64_t *self,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Initialize))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_assemble_m,BHYPRE_IJMATRIXVIEW_ASSEMBLE_M,bHYPRE_IJMatrixView_Assemble_m)
(
  int64_t *self,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Assemble))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
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
SIDLFortran90Symbol(bhypre_ijmatrixview_addref_m,BHYPRE_IJMATRIXVIEW_ADDREF_M,bHYPRE_IJMatrixView_addRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_deleteref_m,BHYPRE_IJMATRIXVIEW_DELETEREF_M,bHYPRE_IJMatrixView_deleteRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self->d_object,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_issame_m,BHYPRE_IJMATRIXVIEW_ISSAME_M,bHYPRE_IJMatrixView_isSame_m)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self->d_object,
      _proxy_iobj,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_istype_m,BHYPRE_IJMATRIXVIEW_ISTYPE_M,bHYPRE_IJMatrixView_isType_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  SIDL_F90_Bool *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self->d_object,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran90Symbol(bhypre_ijmatrixview_getclassinfo_m,BHYPRE_IJMATRIXVIEW_GETCLASSINFO_M,bHYPRE_IJMatrixView_getClassInfo_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_IJMatrixView__epv *_epv = NULL;
  struct bHYPRE_IJMatrixView__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_IJMatrixView__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self->d_object,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_createcol_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_CREATECOL_M,
                  bHYPRE_IJMatrixView__array_createCol_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_createrow_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_CREATEROW_M,
                  bHYPRE_IJMatrixView__array_createRow_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_create1d_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_CREATE1D_M,
                  bHYPRE_IJMatrixView__array_create1d_m)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_create2dcol_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_CREATE2DCOL_M,
                  bHYPRE_IJMatrixView__array_create2dCol_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_create2drow_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_CREATE2DROW_M,
                  bHYPRE_IJMatrixView__array_create2dRow_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_addref_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_ADDREF_M,
                  bHYPRE_IJMatrixView__array_addRef_m)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_deleteref_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_DELETEREF_M,
                  bHYPRE_IJMatrixView__array_deleteRef_m)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get1_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET1_M,
                  bHYPRE_IJMatrixView__array_get1_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get2_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET2_M,
                  bHYPRE_IJMatrixView__array_get2_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get3_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET3_M,
                  bHYPRE_IJMatrixView__array_get3_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get4_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET4_M,
                  bHYPRE_IJMatrixView__array_get4_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get5_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET5_M,
                  bHYPRE_IJMatrixView__array_get5_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get6_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET6_M,
                  bHYPRE_IJMatrixView__array_get6_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get7_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET7_M,
                  bHYPRE_IJMatrixView__array_get7_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_get_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_GET_M,
                  bHYPRE_IJMatrixView__array_get_m)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set1_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET1_M,
                  bHYPRE_IJMatrixView__array_set1_m)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set2_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET2_M,
                  bHYPRE_IJMatrixView__array_set2_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set3_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET3_M,
                  bHYPRE_IJMatrixView__array_set3_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set4_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET4_M,
                  bHYPRE_IJMatrixView__array_set4_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set5_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET5_M,
                  bHYPRE_IJMatrixView__array_set5_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set6_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET6_M,
                  bHYPRE_IJMatrixView__array_set6_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set7_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET7_M,
                  bHYPRE_IJMatrixView__array_set7_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_set_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SET_M,
                  bHYPRE_IJMatrixView__array_set_m)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_dimen_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_DIMEN_M,
                  bHYPRE_IJMatrixView__array_dimen_m)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_lower_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_LOWER_M,
                  bHYPRE_IJMatrixView__array_lower_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_upper_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_UPPER_M,
                  bHYPRE_IJMatrixView__array_upper_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_length_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_LENGTH_M,
                  bHYPRE_IJMatrixView__array_length_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_stride_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_STRIDE_M,
                  bHYPRE_IJMatrixView__array_stride_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_iscolumnorder_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_ISCOLUMNORDER_M,
                  bHYPRE_IJMatrixView__array_isColumnOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_isroworder_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_ISROWORDER_M,
                  bHYPRE_IJMatrixView__array_isRowOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_copy_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_COPY_M,
                  bHYPRE_IJMatrixView__array_copy_m)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_smartcopy_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SMARTCOPY_M,
                  bHYPRE_IJMatrixView__array_smartCopy_m)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran90Symbol(bhypre_ijmatrixview__array_slice_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_SLICE_M,
                  bHYPRE_IJMatrixView__array_slice_m)
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
SIDLFortran90Symbol(bhypre_ijmatrixview__array_ensure_m,
                  BHYPRE_IJMATRIXVIEW__ARRAY_ENSURE_M,
                  bHYPRE_IJMatrixView__array_ensure_m)
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

#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE__IJMatrixView__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE__IJMatrixView__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE__IJMatrixView__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE__IJMatrixView__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 10;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct bHYPRE__IJMatrixView__epv s_rem_epv__bhypre__ijmatrixview;

static struct bHYPRE_IJMatrixView__epv s_rem_epv__bhypre_ijmatrixview;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE__IJMatrixView__cast(
  struct bHYPRE__IJMatrixView__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "bHYPRE.ProblemDefinition");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_bhypre_problemdefinition);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.MatrixVectorView");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_matrixvectorview);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.IJMatrixView");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_ijmatrixview);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseinterface);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE._IJMatrixView");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
        return cast;
      }
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE__IJMatrixView__delete(
  struct bHYPRE__IJMatrixView__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE__IJMatrixView__getURL(
  struct bHYPRE__IJMatrixView__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE__IJMatrixView__raddRef(
  struct bHYPRE__IJMatrixView__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    sidl_BaseInterface throwaway_exception = NULL;
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
      &throwaway_exception);
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_bHYPRE__IJMatrixView__isRemote(
    struct bHYPRE__IJMatrixView__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE__IJMatrixView__set_hooks(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in */ sidl_bool on,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView._set_hooks.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE__IJMatrixView__exec(
  struct bHYPRE__IJMatrixView__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:SetLocalRange */
static int32_t
remote_bHYPRE__IJMatrixView_SetLocalRange(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetLocalRange", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "ilower", ilower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "iupper", iupper, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "jlower", jlower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "jupper", jupper, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.SetLocalRange.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetValues */
static int32_t
remote_bHYPRE__IJMatrixView_SetValues(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* in rarray[nnonzeros] */ struct sidl_double__array* values,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.SetValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:AddToValues */
static int32_t
remote_bHYPRE__IJMatrixView_AddToValues(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* in rarray[nnonzeros] */ struct sidl_double__array* values,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "AddToValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.AddToValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetLocalRange */
static int32_t
remote_bHYPRE__IJMatrixView_GetLocalRange(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetLocalRange", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.GetLocalRange.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackInt( _rsvp, "ilower", ilower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackInt( _rsvp, "iupper", iupper, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackInt( _rsvp, "jlower", jlower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackInt( _rsvp, "jupper", jupper, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetRowCounts */
static int32_t
remote_bHYPRE__IJMatrixView_GetRowCounts(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* inout rarray[nrows] */ struct sidl_int__array** ncols,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetRowCounts", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", *ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.GetRowCounts.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackIntArray( _rsvp, "ncols", ncols,
      sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetValues */
static int32_t
remote_bHYPRE__IJMatrixView_GetValues(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* inout rarray[nnonzeros] */ struct sidl_double__array** values,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", *values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.GetValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDoubleArray( _rsvp, "values", values,
      sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetRowSizes */
static int32_t
remote_bHYPRE__IJMatrixView_SetRowSizes(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in rarray[nrows] */ struct sidl_int__array* sizes,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetRowSizes", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "sizes", sizes,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.SetRowSizes.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Print */
static int32_t
remote_bHYPRE__IJMatrixView_Print(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in */ const char* filename,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Print", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.Print.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Read */
static int32_t
remote_bHYPRE__IJMatrixView_Read(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in */ const char* filename,
  /* in */ struct bHYPRE_MPICommunicator__object* comm,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Read", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename,
      _ex);SIDL_CHECK(*_ex);
    if(comm){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)comm,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "comm", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "comm", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.Read.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE__IJMatrixView_SetCommunicator(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetCommunicator", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(mpi_comm){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)mpi_comm,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "mpi_comm", _url,
        _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "mpi_comm", NULL,
        _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.SetCommunicator.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Destroy */
static void
remote_bHYPRE__IJMatrixView_Destroy(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.Destroy.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:Initialize */
static int32_t
remote_bHYPRE__IJMatrixView_Initialize(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Initialize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.Initialize.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Assemble */
static int32_t
remote_bHYPRE__IJMatrixView_Assemble(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Assemble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.Assemble.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE__IJMatrixView_addRef(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE__IJMatrixView__remote* r_obj = (struct 
      bHYPRE__IJMatrixView__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE__IJMatrixView_deleteRef(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE__IJMatrixView__remote* r_obj = (struct 
      bHYPRE__IJMatrixView__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_bHYPRE__IJMatrixView_isSame(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.isSame.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE__IJMatrixView_isType(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.isType.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_bHYPRE__IJMatrixView_getClassInfo(
  /* in */ struct bHYPRE__IJMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE__IJMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._IJMatrixView.getClassInfo.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE__IJMatrixView__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE__IJMatrixView__epv*     epv = &s_rem_epv__bhypre__ijmatrixview;
  struct bHYPRE_IJMatrixView__epv*      e0  = &s_rem_epv__bhypre_ijmatrixview;
  struct bHYPRE_MatrixVectorView__epv*  e1  = 
    &s_rem_epv__bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__epv* e2  = 
    &s_rem_epv__bhypre_problemdefinition;
  struct sidl_BaseInterface__epv*       e3  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast                = remote_bHYPRE__IJMatrixView__cast;
  epv->f__delete              = remote_bHYPRE__IJMatrixView__delete;
  epv->f__exec                = remote_bHYPRE__IJMatrixView__exec;
  epv->f__getURL              = remote_bHYPRE__IJMatrixView__getURL;
  epv->f__raddRef             = remote_bHYPRE__IJMatrixView__raddRef;
  epv->f__isRemote            = remote_bHYPRE__IJMatrixView__isRemote;
  epv->f__set_hooks           = remote_bHYPRE__IJMatrixView__set_hooks;
  epv->f__ctor                = NULL;
  epv->f__ctor2               = NULL;
  epv->f__dtor                = NULL;
  epv->f_SetLocalRange        = remote_bHYPRE__IJMatrixView_SetLocalRange;
  epv->f_SetValues            = remote_bHYPRE__IJMatrixView_SetValues;
  epv->f_AddToValues          = remote_bHYPRE__IJMatrixView_AddToValues;
  epv->f_GetLocalRange        = remote_bHYPRE__IJMatrixView_GetLocalRange;
  epv->f_GetRowCounts         = remote_bHYPRE__IJMatrixView_GetRowCounts;
  epv->f_GetValues            = remote_bHYPRE__IJMatrixView_GetValues;
  epv->f_SetRowSizes          = remote_bHYPRE__IJMatrixView_SetRowSizes;
  epv->f_Print                = remote_bHYPRE__IJMatrixView_Print;
  epv->f_Read                 = remote_bHYPRE__IJMatrixView_Read;
  epv->f_SetCommunicator      = remote_bHYPRE__IJMatrixView_SetCommunicator;
  epv->f_Destroy              = remote_bHYPRE__IJMatrixView_Destroy;
  epv->f_Initialize           = remote_bHYPRE__IJMatrixView_Initialize;
  epv->f_Assemble             = remote_bHYPRE__IJMatrixView_Assemble;
  epv->f_addRef               = remote_bHYPRE__IJMatrixView_addRef;
  epv->f_deleteRef            = remote_bHYPRE__IJMatrixView_deleteRef;
  epv->f_isSame               = remote_bHYPRE__IJMatrixView_isSame;
  epv->f_isType               = remote_bHYPRE__IJMatrixView_isType;
  epv->f_getClassInfo         = remote_bHYPRE__IJMatrixView_getClassInfo;

  e0->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_SetLocalRange   = (int32_t (*)(void*,int32_t,int32_t,int32_t,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetLocalRange;
  e0->f_SetValues       = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetValues;
  e0->f_AddToValues     = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_AddToValues;
  e0->f_GetLocalRange   = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
    int32_t*,struct sidl_BaseInterface__object **)) epv->f_GetLocalRange;
  e0->f_GetRowCounts    = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array**,
    struct sidl_BaseInterface__object **)) epv->f_GetRowCounts;
  e0->f_GetValues       = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,struct sidl_double__array**,
    struct sidl_BaseInterface__object **)) epv->f_GetValues;
  e0->f_SetRowSizes     = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetRowSizes;
  e0->f_Print           = (int32_t (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_Print;
  e0->f_Read            = (int32_t (*)(void*,const char*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_Read;
  e0->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e0->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e0->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e0->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e1->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e1->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e1->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e2->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e2->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e2->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e2->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e3->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e3->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e3->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e3->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e3->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE__IJMatrixView__object* self;

  struct bHYPRE__IJMatrixView__object* s0;

  struct bHYPRE__IJMatrixView__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex);
    if(ar) {
      sidl_BaseInterface_addRef(bi, _ex);
    }
    return bHYPRE_IJMatrixView__rmicast(bi, _ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE__IJMatrixView__object*) malloc(
      sizeof(struct bHYPRE__IJMatrixView__object));

  r_obj =
    (struct bHYPRE__IJMatrixView__remote*) malloc(
      sizeof(struct bHYPRE__IJMatrixView__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE__IJMatrixView__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_bhypre_ijmatrixview.d_epv    = &s_rem_epv__bhypre_ijmatrixview;
  s0->d_bhypre_ijmatrixview.d_object = (void*) self;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre__ijmatrixview;

  self->d_data = (void*) r_obj;

  return bHYPRE_IJMatrixView__rmicast(self, _ex);
}
/* Create an instance that uses an already existing  */
/* InstanceHandel to connect to an existing remote object. */
static struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE__IJMatrixView__object* self;

  struct bHYPRE__IJMatrixView__object* s0;

  struct bHYPRE__IJMatrixView__remote* r_obj;
  self =
    (struct bHYPRE__IJMatrixView__object*) malloc(
      sizeof(struct bHYPRE__IJMatrixView__object));

  r_obj =
    (struct bHYPRE__IJMatrixView__remote*) malloc(
      sizeof(struct bHYPRE__IJMatrixView__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE__IJMatrixView__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_bhypre_ijmatrixview.d_epv    = &s_rem_epv__bhypre_ijmatrixview;
  s0->d_bhypre_ijmatrixview.d_object = (void*) self;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre__ijmatrixview;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance, _ex);
  return bHYPRE_IJMatrixView__rmicast(self, _ex);
}
/*
 * Cast method for interface and class type conversions.
 */

struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_IJMatrixView__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.IJMatrixView",
      (void*)bHYPRE_IJMatrixView__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_IJMatrixView__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.IJMatrixView", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_IJMatrixView__object*
bHYPRE_IJMatrixView__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return bHYPRE_IJMatrixView__remoteConnect(url, ar, _ex);
}

