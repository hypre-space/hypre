/*
 * File:          bHYPRE_IJParCSRMatrix_Skel.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_IJParCSRMatrix_IOR.h"
#include "bHYPRE_IJParCSRMatrix.h"
#include <stddef.h>

extern
void
impl_bHYPRE_IJParCSRMatrix__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRMatrix__ctor(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRMatrix__ctor2(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_IJParCSRMatrix__dtor(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_IJParCSRMatrix
impl_bHYPRE_IJParCSRMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_IJParCSRMatrix
impl_bHYPRE_IJParCSRMatrix_GenerateLaplacian(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t nx,
  /* in */ int32_t ny,
  /* in */ int32_t nz,
  /* in */ int32_t Px,
  /* in */ int32_t Py,
  /* in */ int32_t Pz,
  /* in */ int32_t p,
  /* in */ int32_t q,
  /* in */ int32_t r,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* in */ int32_t discretization,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_CoefficientAccess(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJMatrixView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJMatrixView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[local_nrows] */ int32_t* diag_sizes,
  /* in rarray[local_nrows] */ int32_t* offdiag_sizes,
  /* in */ int32_t local_nrows,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_AddToValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* in rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetLocalRange(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ int32_t* ilower,
  /* out */ int32_t* iupper,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* rows,
  /* inout rarray[nrows] */ int32_t* ncols,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t nrows,
  /* in rarray[nrows] */ int32_t* ncols,
  /* in rarray[nrows] */ int32_t* rows,
  /* in rarray[nnonzeros] */ int32_t* cols,
  /* inout rarray[nnonzeros] */ double* values,
  /* in */ int32_t nnonzeros,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ int32_t* sizes,
  /* in */ int32_t nrows,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Print(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Read(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Initialize(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Assemble(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Setup(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_Apply(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_ApplyAdjoint(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_IJParCSRMatrix_GetRow(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out array<int,column-major> */ struct sidl_int__array** col_ind,
  /* out array<double,column-major> */ struct sidl_double__array** values,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_CoefficientAccess__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_CoefficientAccess(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJMatrixView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJMatrixView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJMatrixView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_IJParCSRMatrix__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_IJParCSRMatrix_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
static bHYPRE_IJParCSRMatrix
skel_bHYPRE_IJParCSRMatrix_GenerateLaplacian(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t nx,
  /* in */ int32_t ny,
  /* in */ int32_t nz,
  /* in */ int32_t Px,
  /* in */ int32_t Py,
  /* in */ int32_t Pz,
  /* in */ int32_t p,
  /* in */ int32_t q,
  /* in */ int32_t r,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
  /* in */ int32_t discretization,
/* out */ sidl_BaseInterface *_ex)
{
  bHYPRE_IJParCSRMatrix _return;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nvalues = sidlLength(values_proxy,0);
  _return =
    impl_bHYPRE_IJParCSRMatrix_GenerateLaplacian(
      mpi_comm,
      nx,
      ny,
      nz,
      Px,
      Py,
      Pz,
      p,
      q,
      r,
      values_tmp,
      nvalues,
      discretization,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[local_nrows] */ struct sidl_int__array* diag_sizes,
  /* in rarray[local_nrows] */ struct sidl_int__array* offdiag_sizes,
/* out */ sidl_BaseInterface *_ex)
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
      local_nrows,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* in rarray[nnonzeros] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
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
      nnonzeros,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_AddToValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* in rarray[nnonzeros] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
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
      nnonzeros,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetRowCounts(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* inout rarray[nrows] */ struct sidl_int__array** ncols,
/* out */ sidl_BaseInterface *_ex)
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
      ncols_tmp,
      _ex);
  sidl_int__array_init(ncols_tmp, *ncols, 1, (*ncols)->d_metadata.d_lower,
    (*ncols)->d_metadata.d_upper, (*ncols)->d_metadata.d_stride);

  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetValues(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ struct sidl_int__array* ncols,
  /* in rarray[nrows] */ struct sidl_int__array* rows,
  /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
  /* inout rarray[nnonzeros] */ struct sidl_double__array** values,
/* out */ sidl_BaseInterface *_ex)
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
      nnonzeros,
      _ex);
  sidl_double__array_init(values_tmp, *values, 1, (*values)->d_metadata.d_lower,
    (*values)->d_metadata.d_upper, (*values)->d_metadata.d_stride);

  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in rarray[nrows] */ struct sidl_int__array* sizes,
/* out */ sidl_BaseInterface *_ex)
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
      nrows,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ struct sidl_int__array* value,
/* out */ sidl_BaseInterface *_ex)
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
      nvalues,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy,
      _ex);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ struct sidl_double__array* value,
/* out */ sidl_BaseInterface *_ex)
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
      nvalues,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2,
    sidl_column_major_order);
  _return =
    impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy,
      _ex);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_IJParCSRMatrix_GetRow(
  /* in */ bHYPRE_IJParCSRMatrix self,
  /* in */ int32_t row,
  /* out */ int32_t* size,
  /* out array<int,column-major> */ struct sidl_int__array** col_ind,
  /* out array<double,column-major> */ struct sidl_double__array** values,
/* out */ sidl_BaseInterface *_ex)
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
      &values_proxy,
      _ex);
  *col_ind = sidl_int__array_ensure(col_ind_proxy, 1, sidl_column_major_order);
  sidl_int__array_deleteRef(col_ind_proxy);
  *values = sidl_double__array_ensure(values_proxy, 1, sidl_column_major_order);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_IJParCSRMatrix__set_epv(struct bHYPRE_IJParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_IJParCSRMatrix__ctor;
  epv->f__ctor2 = impl_bHYPRE_IJParCSRMatrix__ctor2;
  epv->f__dtor = impl_bHYPRE_IJParCSRMatrix__dtor;
  epv->f_SetDiagOffdSizes = skel_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes;
  epv->f_SetLocalRange = impl_bHYPRE_IJParCSRMatrix_SetLocalRange;
  epv->f_SetValues = skel_bHYPRE_IJParCSRMatrix_SetValues;
  epv->f_AddToValues = skel_bHYPRE_IJParCSRMatrix_AddToValues;
  epv->f_GetLocalRange = impl_bHYPRE_IJParCSRMatrix_GetLocalRange;
  epv->f_GetRowCounts = skel_bHYPRE_IJParCSRMatrix_GetRowCounts;
  epv->f_GetValues = skel_bHYPRE_IJParCSRMatrix_GetValues;
  epv->f_SetRowSizes = skel_bHYPRE_IJParCSRMatrix_SetRowSizes;
  epv->f_Print = impl_bHYPRE_IJParCSRMatrix_Print;
  epv->f_Read = impl_bHYPRE_IJParCSRMatrix_Read;
  epv->f_SetCommunicator = impl_bHYPRE_IJParCSRMatrix_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_IJParCSRMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_IJParCSRMatrix_Assemble;
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
  epv->f_ApplyAdjoint = impl_bHYPRE_IJParCSRMatrix_ApplyAdjoint;
  epv->f_GetRow = skel_bHYPRE_IJParCSRMatrix_GetRow;

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
  sepv->f_GenerateLaplacian = skel_bHYPRE_IJParCSRMatrix_GenerateLaplacian;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_IJParCSRMatrix__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_IJParCSRMatrix__load(&_throwaway_exception);
}
struct bHYPRE_CoefficientAccess__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_CoefficientAccess(url, ar,
    _ex);
}

struct bHYPRE_CoefficientAccess__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_CoefficientAccess(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_CoefficientAccess(bi, _ex);
}

struct bHYPRE_IJMatrixView__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJMatrixView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJMatrixView(url, ar, _ex);
}

struct bHYPRE_IJMatrixView__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJMatrixView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJMatrixView(bi, _ex);
}

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_IJParCSRMatrix(url, ar,
    _ex);
}

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJParCSRMatrix(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_IJParCSRMatrix(bi, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MPICommunicator(url, ar,
    _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(url, ar,
    _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_MatrixVectorView(bi, _ex);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Operator(url, ar, _ex);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Operator(bi, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(url, ar,
    _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_ProblemDefinition(bi, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_bHYPRE_Vector(url, ar, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_bHYPRE_Vector(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_IJParCSRMatrix_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fconnect_sidl_RuntimeException(url, ar,
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_IJParCSRMatrix_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_IJParCSRMatrix_fcast_sidl_RuntimeException(bi, _ex);
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
