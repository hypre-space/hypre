/*
 * File:          bHYPRE_StructMatrix_Skel.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side glue code for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_StructMatrix_IOR.h"
#include "bHYPRE_StructMatrix.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructMatrix__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructMatrix__ctor(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructMatrix__ctor2(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructMatrix__dtor(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_StructMatrix
impl_bHYPRE_StructMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructGrid grid,
  /* in */ bHYPRE_StructStencil stencil,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_Operator(void* bi, sidl_BaseInterface* 
  _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructGrid(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrix(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructMatrixView__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrixView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructMatrixView__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrixView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructStencil(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_StructStencil stencil,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetConstantEntries(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t num_stencil_constant_points,
  /* in rarray[num_stencil_constant_points] */ int32_t* stencil_constant_points,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetConstantValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructMatrix_Destroy(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_Initialize(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_Assemble(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_Setup(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_Apply(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructMatrix_ApplyAdjoint(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_Operator(void* bi, sidl_BaseInterface* 
  _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructGrid(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrix(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructMatrixView__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrixView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructMatrixView__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrixView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructStencil(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_Vector(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_StructMatrix_SetValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in rarray[num_stencil_indices] */ struct sidl_int__array* stencil_indices,
  /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1, 
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_int__array* stencil_indices_proxy = sidl_int__array_ensure(
    stencil_indices, 1, sidl_column_major_order);
  int32_t* stencil_indices_tmp = stencil_indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t num_stencil_indices = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetValues(
      self,
      index_tmp,
      dim,
      num_stencil_indices,
      stencil_indices_tmp,
      values_tmp,
      _ex);
  sidl_int__array_deleteRef(index_proxy);
  sidl_int__array_deleteRef(stencil_indices_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetBoxValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in rarray[num_stencil_indices] */ struct sidl_int__array* stencil_indices,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1, 
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1, 
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  struct sidl_int__array* stencil_indices_proxy = sidl_int__array_ensure(
    stencil_indices, 1, sidl_column_major_order);
  int32_t* stencil_indices_tmp = stencil_indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t num_stencil_indices = sidlLength(stencil_indices_proxy,0);
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetBoxValues(
      self,
      ilower_tmp,
      iupper_tmp,
      dim,
      num_stencil_indices,
      stencil_indices_tmp,
      values_tmp,
      nvalues,
      _ex);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_int__array_deleteRef(stencil_indices_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetNumGhost(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  int32_t* num_ghost_tmp = num_ghost_proxy->d_firstElement;
  int32_t dim2 = sidlLength(num_ghost_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetNumGhost(
      self,
      num_ghost_tmp,
      dim2,
      _ex);
  sidl_int__array_deleteRef(num_ghost_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetConstantEntries(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[num_stencil_constant_points] */ struct sidl_int__array* 
    stencil_constant_points,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* stencil_constant_points_proxy = 
    sidl_int__array_ensure(stencil_constant_points, 1, sidl_column_major_order);
  int32_t* stencil_constant_points_tmp = 
    stencil_constant_points_proxy->d_firstElement;
  int32_t num_stencil_constant_points = sidlLength(
    stencil_constant_points_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetConstantEntries(
      self,
      num_stencil_constant_points,
      stencil_constant_points_tmp,
      _ex);
  sidl_int__array_deleteRef(stencil_constant_points_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetConstantValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[num_stencil_indices] */ struct sidl_int__array* stencil_indices,
  /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* stencil_indices_proxy = sidl_int__array_ensure(
    stencil_indices, 1, sidl_column_major_order);
  int32_t* stencil_indices_tmp = stencil_indices_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t num_stencil_indices = sidlLength(values_proxy,0);
  _return =
    impl_bHYPRE_StructMatrix_SetConstantValues(
      self,
      num_stencil_indices,
      stencil_indices_tmp,
      values_tmp,
      _ex);
  sidl_int__array_deleteRef(stencil_indices_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
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
    impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues,
      _ex);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2, 
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy,
      _ex);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
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
    impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues,
      _ex);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2, 
    sidl_column_major_order);
  _return =
    impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
      self,
      name,
      value_proxy,
      _ex);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructMatrix__set_epv(struct bHYPRE_StructMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructMatrix__ctor;
  epv->f__ctor2 = impl_bHYPRE_StructMatrix__ctor2;
  epv->f__dtor = impl_bHYPRE_StructMatrix__dtor;
  epv->f_SetGrid = impl_bHYPRE_StructMatrix_SetGrid;
  epv->f_SetStencil = impl_bHYPRE_StructMatrix_SetStencil;
  epv->f_SetValues = skel_bHYPRE_StructMatrix_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_StructMatrix_SetBoxValues;
  epv->f_SetNumGhost = skel_bHYPRE_StructMatrix_SetNumGhost;
  epv->f_SetSymmetric = impl_bHYPRE_StructMatrix_SetSymmetric;
  epv->f_SetConstantEntries = skel_bHYPRE_StructMatrix_SetConstantEntries;
  epv->f_SetConstantValues = skel_bHYPRE_StructMatrix_SetConstantValues;
  epv->f_SetCommunicator = impl_bHYPRE_StructMatrix_SetCommunicator;
  epv->f_Destroy = impl_bHYPRE_StructMatrix_Destroy;
  epv->f_Initialize = impl_bHYPRE_StructMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructMatrix_Assemble;
  epv->f_SetIntParameter = impl_bHYPRE_StructMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_bHYPRE_StructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_bHYPRE_StructMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = skel_bHYPRE_StructMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = skel_bHYPRE_StructMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_StructMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_StructMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_StructMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_StructMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_StructMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_StructMatrix_Apply;
  epv->f_ApplyAdjoint = impl_bHYPRE_StructMatrix_ApplyAdjoint;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructMatrix__set_sepv(struct bHYPRE_StructMatrix__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_StructMatrix_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_StructMatrix__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_StructMatrix__load(&_throwaway_exception);
}
struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MPICommunicator(url, ar, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructMatrix_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MatrixVectorView(url, ar, 
    _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_StructMatrix_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_MatrixVectorView(bi, _ex);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(url, ar, _ex);
}

struct bHYPRE_Operator__object* skel_bHYPRE_StructMatrix_fcast_bHYPRE_Operator(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_Operator(bi, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(url, ar, 
    _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_StructMatrix_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_ProblemDefinition(bi, _ex);
}

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(url, ar, _ex);
}

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructGrid(void* bi, sidl_BaseInterface 
  *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructGrid(bi, _ex);
}

struct bHYPRE_StructMatrix__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(url, ar, _ex);
}

struct bHYPRE_StructMatrix__object* 
  skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrix(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrix(bi, _ex);
}

struct bHYPRE_StructMatrixView__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrixView(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrixView(url, ar, 
    _ex);
}

struct bHYPRE_StructMatrixView__object* 
  skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrixView(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrixView(bi, _ex);
}

struct bHYPRE_StructStencil__object* 
  skel_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(url, ar, _ex);
}

struct bHYPRE_StructStencil__object* 
  skel_bHYPRE_StructMatrix_fcast_bHYPRE_StructStencil(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructStencil(bi, _ex);
}

struct bHYPRE_Vector__object* skel_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(url, ar, _ex);
}

struct bHYPRE_Vector__object* skel_bHYPRE_StructMatrix_fcast_bHYPRE_Vector(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_bHYPRE_Vector(bi, _ex);
}

struct sidl_BaseClass__object* skel_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* skel_bHYPRE_StructMatrix_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructMatrix_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* skel_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* skel_bHYPRE_StructMatrix_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructMatrix_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructMatrix_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructMatrix_fcast_sidl_RuntimeException(bi, _ex);
}

struct bHYPRE_StructMatrix__data*
bHYPRE_StructMatrix__get_data(bHYPRE_StructMatrix self)
{
  return (struct bHYPRE_StructMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructMatrix__set_data(
  bHYPRE_StructMatrix self,
  struct bHYPRE_StructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
