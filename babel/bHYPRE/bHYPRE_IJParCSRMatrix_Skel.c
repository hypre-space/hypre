/*
 * File:          bHYPRE_IJParCSRMatrix_Skel.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:32 PST
 * Description:   Server-side glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 789
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_IJParCSRMatrix_IOR.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include <stddef.h>

extern void
impl_bHYPRE_IJParCSRMatrix__ctor(
  bHYPRE_IJParCSRMatrix);

extern void
impl_bHYPRE_IJParCSRMatrix__dtor(
  bHYPRE_IJParCSRMatrix);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  bHYPRE_IJParCSRMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_GetRow(
  bHYPRE_IJParCSRMatrix,
  int32_t,
  int32_t*,
  struct SIDL_int__array**,
  struct SIDL_double__array**);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetCommunicator(
  bHYPRE_IJParCSRMatrix,
  void*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_Initialize(
  bHYPRE_IJParCSRMatrix);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_Assemble(
  bHYPRE_IJParCSRMatrix);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_GetObject(
  bHYPRE_IJParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetLocalRange(
  bHYPRE_IJParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetValues(
  bHYPRE_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_AddToValues(
  bHYPRE_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_GetLocalRange(
  bHYPRE_IJParCSRMatrix,
  int32_t*,
  int32_t*,
  int32_t*,
  int32_t*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
  bHYPRE_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array**);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_GetValues(
  bHYPRE_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array**);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
  bHYPRE_IJParCSRMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_Print(
  bHYPRE_IJParCSRMatrix,
  const char*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_Read(
  bHYPRE_IJParCSRMatrix,
  const char*,
  void*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntParameter(
  bHYPRE_IJParCSRMatrix,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  bHYPRE_IJParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetStringParameter(
  bHYPRE_IJParCSRMatrix,
  const char*,
  const char*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  bHYPRE_IJParCSRMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  bHYPRE_IJParCSRMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  bHYPRE_IJParCSRMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  bHYPRE_IJParCSRMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_GetIntValue(
  bHYPRE_IJParCSRMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_GetDoubleValue(
  bHYPRE_IJParCSRMatrix,
  const char*,
  double*);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_Setup(
  bHYPRE_IJParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_IJParCSRMatrix_Apply(
  bHYPRE_IJParCSRMatrix,
  bHYPRE_Vector,
  bHYPRE_Vector*);

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  bHYPRE_IJParCSRMatrix self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes)
{
  int32_t _return;
  struct SIDL_int__array* diag_sizes_proxy = SIDL_int__array_ensure(diag_sizes,
    1, SIDL_column_major_order);
  struct SIDL_int__array* offdiag_sizes_proxy = 
    SIDL_int__array_ensure(offdiag_sizes, 1, SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
      self,
      diag_sizes_proxy,
      offdiag_sizes_proxy);
  SIDL_int__array_deleteRef(diag_sizes_proxy);
  SIDL_int__array_deleteRef(offdiag_sizes_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetRow(
  bHYPRE_IJParCSRMatrix self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  int32_t _return;
  struct SIDL_int__array* col_ind_proxy = NULL;
  struct SIDL_double__array* values_proxy = NULL;
  _return =
    impl_bHYPRE_IJParCSRMatrix_GetRow(
      self,
      row,
      size,
      &col_ind_proxy,
      &values_proxy);
  *col_ind = SIDL_int__array_ensure(col_ind_proxy, 1, SIDL_column_major_order);
  SIDL_int__array_deleteRef(col_ind_proxy);
  *values = SIDL_double__array_ensure(values_proxy, 1, SIDL_column_major_order);
  SIDL_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetValues(
  bHYPRE_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  int32_t _return;
  struct SIDL_int__array* ncols_proxy = SIDL_int__array_ensure(ncols, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* rows_proxy = SIDL_int__array_ensure(rows, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* cols_proxy = SIDL_int__array_ensure(cols, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* values_proxy = SIDL_double__array_ensure(values, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetValues(
      self,
      nrows,
      ncols_proxy,
      rows_proxy,
      cols_proxy,
      values_proxy);
  SIDL_int__array_deleteRef(ncols_proxy);
  SIDL_int__array_deleteRef(rows_proxy);
  SIDL_int__array_deleteRef(cols_proxy);
  SIDL_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_AddToValues(
  bHYPRE_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  int32_t _return;
  struct SIDL_int__array* ncols_proxy = SIDL_int__array_ensure(ncols, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* rows_proxy = SIDL_int__array_ensure(rows, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* cols_proxy = SIDL_int__array_ensure(cols, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* values_proxy = SIDL_double__array_ensure(values, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_AddToValues(
      self,
      nrows,
      ncols_proxy,
      rows_proxy,
      cols_proxy,
      values_proxy);
  SIDL_int__array_deleteRef(ncols_proxy);
  SIDL_int__array_deleteRef(rows_proxy);
  SIDL_int__array_deleteRef(cols_proxy);
  SIDL_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetRowCounts(
  bHYPRE_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* rows,
  struct SIDL_int__array** ncols)
{
  int32_t _return;
  struct SIDL_int__array* rows_proxy = SIDL_int__array_ensure(rows, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* ncols_proxy = SIDL_int__array_ensure(*ncols, 1,
    SIDL_column_major_order);
  SIDL_int__array_deleteRef(*ncols);
  _return =
    impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
      self,
      nrows,
      rows_proxy,
      &ncols_proxy);
  SIDL_int__array_deleteRef(rows_proxy);
  *ncols = SIDL_int__array_ensure(ncols_proxy, 1, SIDL_column_major_order);
  SIDL_int__array_deleteRef(ncols_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetValues(
  bHYPRE_IJParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array** values)
{
  int32_t _return;
  struct SIDL_int__array* ncols_proxy = SIDL_int__array_ensure(ncols, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* rows_proxy = SIDL_int__array_ensure(rows, 1,
    SIDL_column_major_order);
  struct SIDL_int__array* cols_proxy = SIDL_int__array_ensure(cols, 1,
    SIDL_column_major_order);
  struct SIDL_double__array* values_proxy = SIDL_double__array_ensure(*values,
    1, SIDL_column_major_order);
  SIDL_double__array_deleteRef(*values);
  _return =
    impl_bHYPRE_IJParCSRMatrix_GetValues(
      self,
      nrows,
      ncols_proxy,
      rows_proxy,
      cols_proxy,
      &values_proxy);
  SIDL_int__array_deleteRef(ncols_proxy);
  SIDL_int__array_deleteRef(rows_proxy);
  SIDL_int__array_deleteRef(cols_proxy);
  *values = SIDL_double__array_ensure(values_proxy, 1, SIDL_column_major_order);
  SIDL_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetRowSizes(
  bHYPRE_IJParCSRMatrix self,
  struct SIDL_int__array* sizes)
{
  int32_t _return;
  struct SIDL_int__array* sizes_proxy = SIDL_int__array_ensure(sizes, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
      self,
      sizes_proxy);
  SIDL_int__array_deleteRef(sizes_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value)
{
  int32_t _return;
  struct SIDL_int__array* value_proxy = SIDL_int__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
      self,
      name,
      value_proxy);
  SIDL_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_int__array* value)
{
  int32_t _return;
  struct SIDL_int__array* value_proxy = SIDL_int__array_ensure(value, 2,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy);
  SIDL_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 1,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  bHYPRE_IJParCSRMatrix self,
  const char* name,
  struct SIDL_double__array* value)
{
  int32_t _return;
  struct SIDL_double__array* value_proxy = SIDL_double__array_ensure(value, 2,
    SIDL_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy);
  SIDL_double__array_deleteRef(value_proxy);
  return _return;
}

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
