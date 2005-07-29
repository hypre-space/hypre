/*
 * File:          bHYPRE_IJParCSRMatrix_Skel.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_IJParCSRMatrix_IOR.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include <stddef.h>

extern
void
impl_bHYPRE_IJParCSRMatrix__load(
  void);

extern
void
impl_bHYPRE_IJParCSRMatrix__ctor(
  /* in */ bHYPRE_IJParCSRMatrix self);

extern
void
impl_bHYPRE_IJParCSRMatrix__dtor(
  /* in */ bHYPRE_IJParCSRMatrix self);

extern
bHYPRE_IJParCSRMatrix
impl_bHYPRE_IJParCSRMatrix_Create(
  /* in */ void* mpi_comm,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_CoefficientAccess(struct 
  bHYPRE_CoefficientAccess__object* obj);
extern struct bHYPRE_IJBuildMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJBuildMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJBuildMatrix(struct 
  bHYPRE_IJBuildMatrix__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t* diag_sizes,
  /* in */ int32_t* offdiag_sizes,
  /* in */ int32_t local_nrows);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRow(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out */ struct sidl_int__array** col_ind,
  /* out */ struct sidl_double__array** values);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ void* mpi_comm);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Initialize(
  /* in */ bHYPRE_IJParCSRMatrix self);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Assemble(
  /* in */ bHYPRE_IJParCSRMatrix self);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetObject(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface* A);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* ncols,
  /* in */ int32_t* rows,
  /* in */ int32_t* cols,
  /* in */ double* values,
  /* in */ int32_t nnonzeros);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_AddToValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* ncols,
  /* in */ int32_t* rows,
  /* in */ int32_t* cols,
  /* in */ double* values,
  /* in */ int32_t nnonzeros);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* rows,
  /* inout */ int32_t* ncols);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in */ int32_t* ncols,
  /* in */ int32_t* rows,
  /* in */ int32_t* cols,
  /* inout */ double* values,
  /* in */ int32_t nnonzeros);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t* sizes,
  /* in */ int32_t nrows);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Print(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Read(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ void* comm);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Setup(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Apply(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x);

extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_CoefficientAccess(struct 
  bHYPRE_CoefficientAccess__object* obj);
extern struct bHYPRE_IJBuildMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJBuildMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJBuildMatrix(struct 
  bHYPRE_IJBuildMatrix__object* obj);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ struct sidl_int__array* diag_sizes,
/* in */ struct sidl_int__array* offdiag_sizes)
{
  int32_t _return;
  struct sidl_int__array* diag_sizes_proxy = sidl_int__array_ensure(diag_sizes,
    1, sidl_column_major_order);
  int32_t* diag_sizes_tmp = diag_sizes_proxy->d_firstElement;
  struct sidl_int__array* offdiag_sizes_proxy = 
    sidl_int__array_ensure(offdiag_sizes, 1, sidl_column_major_order);
  int32_t* offdiag_sizes_tmp = offdiag_sizes_proxy->d_firstElement;
  int32_t local_nrows = sidlLength(offdiag_sizes_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
      self,
      diag_sizes_tmp,
      offdiag_sizes_tmp,
      local_nrows);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetRow(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out */ struct sidl_int__array** col_ind,
/* out */ struct sidl_double__array** values)
{
  int32_t _return;
  struct sidl_int__array* col_ind_proxy = NULL;
  struct sidl_double__array* values_proxy = NULL;
  _return =
    impl_bHYPRE_IJParCSRMatrix_GetRow(
      self,
      row,
      size,
      &col_ind_proxy,
      &values_proxy);
  *col_ind = sidl_int__array_ensure(col_ind_proxy, 1, sidl_column_major_order);
  sidl_int__array_deleteRef(col_ind_proxy);
  *values = sidl_double__array_ensure(values_proxy, 1, sidl_column_major_order);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ struct sidl_int__array* ncols,
  /* in */ struct sidl_int__array* rows,
  /* in */ struct sidl_int__array* cols,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ncols_proxy = sidl_int__array_ensure(ncols, 1,
    sidl_column_major_order);
  int32_t* ncols_tmp = ncols_proxy->d_firstElement;
  struct sidl_int__array* rows_proxy = sidl_int__array_ensure(rows, 1,
    sidl_column_major_order);
  int32_t* rows_tmp = rows_proxy->d_firstElement;
  struct sidl_int__array* cols_proxy = sidl_int__array_ensure(cols, 1,
    sidl_column_major_order);
  int32_t* cols_tmp = cols_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nnonzeros = sidlLength(values_proxy,0);
  int32_t nrows = sidlLength(rows_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetValues(
      self,
      nrows,
      ncols_tmp,
      rows_tmp,
      cols_tmp,
      values_tmp,
      nnonzeros);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_AddToValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ struct sidl_int__array* ncols,
  /* in */ struct sidl_int__array* rows,
  /* in */ struct sidl_int__array* cols,
/* in */ struct sidl_double__array* values)
{
  int32_t _return;
  struct sidl_int__array* ncols_proxy = sidl_int__array_ensure(ncols, 1,
    sidl_column_major_order);
  int32_t* ncols_tmp = ncols_proxy->d_firstElement;
  struct sidl_int__array* rows_proxy = sidl_int__array_ensure(rows, 1,
    sidl_column_major_order);
  int32_t* rows_tmp = rows_proxy->d_firstElement;
  struct sidl_int__array* cols_proxy = sidl_int__array_ensure(cols, 1,
    sidl_column_major_order);
  int32_t* cols_tmp = cols_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nnonzeros = sidlLength(values_proxy,0);
  int32_t nrows = sidlLength(rows_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_AddToValues(
      self,
      nrows,
      ncols_tmp,
      rows_tmp,
      cols_tmp,
      values_tmp,
      nnonzeros);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetRowCounts(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ struct sidl_int__array* rows,
/* inout */ struct sidl_int__array** ncols)
{
  int32_t _return;
  struct sidl_int__array* rows_proxy = sidl_int__array_ensure(rows, 1,
    sidl_column_major_order);
  int32_t* rows_tmp = rows_proxy->d_firstElement;
  struct sidl_int__array* ncols_proxy = sidl_int__array_ensure(*ncols, 1,
    sidl_column_major_order);
  int32_t* ncols_tmp = ncols_proxy->d_firstElement;
  int32_t nrows = sidlLength(rows_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
      self,
      nrows,
      rows_tmp,
      ncols_tmp);
  sidl_int__array_init(ncols_tmp, *ncols, 1, (*ncols)->d_metadata.d_lower,
    (*ncols)->d_metadata.d_upper, (*ncols)->d_metadata.d_stride);

  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ struct sidl_int__array* ncols,
  /* in */ struct sidl_int__array* rows,
  /* in */ struct sidl_int__array* cols,
/* inout */ struct sidl_double__array** values)
{
  int32_t _return;
  struct sidl_int__array* ncols_proxy = sidl_int__array_ensure(ncols, 1,
    sidl_column_major_order);
  int32_t* ncols_tmp = ncols_proxy->d_firstElement;
  struct sidl_int__array* rows_proxy = sidl_int__array_ensure(rows, 1,
    sidl_column_major_order);
  int32_t* rows_tmp = rows_proxy->d_firstElement;
  struct sidl_int__array* cols_proxy = sidl_int__array_ensure(cols, 1,
    sidl_column_major_order);
  int32_t* cols_tmp = cols_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(*values,
    1, sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nnonzeros = sidlLength(values_proxy,0);
  int32_t nrows = sidlLength(rows_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_GetValues(
      self,
      nrows,
      ncols_tmp,
      rows_tmp,
      cols_tmp,
      values_tmp,
      nnonzeros);
  sidl_double__array_init(values_tmp, *values, 1, (*values)->d_metadata.d_lower,
    (*values)->d_metadata.d_upper, (*values)->d_metadata.d_stride);

  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
/* in */ struct sidl_int__array* sizes)
{
  int32_t _return;
  struct sidl_int__array* sizes_proxy = sidl_int__array_ensure(sizes, 1,
    sidl_column_major_order);
  int32_t* sizes_tmp = sizes_proxy->d_firstElement;
  int32_t nrows = sidlLength(sizes_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
      self,
      sizes_tmp,
      nrows);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 1,
    sidl_column_major_order);
  int32_t* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_int__array* value)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 1,
    sidl_column_major_order);
  double* value_tmp = value_proxy->d_firstElement;
  int32_t nvalues = sidlLength(value_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
/* in */ struct sidl_double__array* value)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_IJParCSRMatrix__set_epv(struct bHYPRE_IJParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_IJParCSRMatrix__ctor;
  epv->f__dtor = impl_bHYPRE_IJParCSRMatrix__dtor;
  epv->f_SetDiagOffdSizes = skel_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes;
  epv->f_GetRow = skel_bHYPRE_IJParCSRMatrix_GetRow;
  epv->f_SetCommunicator = impl_bHYPRE_IJParCSRMatrix_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_IJParCSRMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_IJParCSRMatrix_Assemble;
  epv->f_GetObject = impl_bHYPRE_IJParCSRMatrix_GetObject;
  epv->f_SetLocalRange = impl_bHYPRE_IJParCSRMatrix_SetLocalRange;
  epv->f_SetValues = skel_bHYPRE_IJParCSRMatrix_SetValues;
  epv->f_AddToValues = skel_bHYPRE_IJParCSRMatrix_AddToValues;
  epv->f_GetLocalRange = impl_bHYPRE_IJParCSRMatrix_GetLocalRange;
  epv->f_GetRowCounts = skel_bHYPRE_IJParCSRMatrix_GetRowCounts;
  epv->f_GetValues = skel_bHYPRE_IJParCSRMatrix_GetValues;
  epv->f_SetRowSizes = skel_bHYPRE_IJParCSRMatrix_SetRowSizes;
  epv->f_Print = impl_bHYPRE_IJParCSRMatrix_Print;
  epv->f_Read = impl_bHYPRE_IJParCSRMatrix_Read;
  epv->f_SetIntParameter = impl_bHYPRE_IJParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_IJParCSRMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_IJParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_IJParCSRMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_IJParCSRMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_IJParCSRMatrix_Apply;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_IJParCSRMatrix__set_sepv(struct bHYPRE_IJParCSRMatrix__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_IJParCSRMatrix_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_IJParCSRMatrix__call_load(void) { 
  impl_bHYPRE_IJParCSRMatrix__load();
}
struct bHYPRE_CoefficientAccess__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_CoefficientAccess(struct 
  bHYPRE_CoefficientAccess__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_CoefficientAccess(obj);
}

struct bHYPRE_IJBuildMatrix__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJBuildMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJBuildMatrix(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJBuildMatrix(struct 
  bHYPRE_IJBuildMatrix__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJBuildMatrix(obj);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Operator(obj);
}

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_IJParCSRMatrix(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_Vector(obj);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_bHYPRE_ProblemDefinition(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_IJParCSRMatrix_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_IJParCSRMatrix__data*
bHYPRE_IJParCSRMatrix__get_data(bHYPRE_IJParCSRMatrix self)
{
  return (struct bHYPRE_IJParCSRMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_IJParCSRMatrix__set_data(
  bHYPRE_IJParCSRMatrix self,
  struct bHYPRE_IJParCSRMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
