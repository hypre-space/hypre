/*
 * File:          bHYPRE_SStructParCSRMatrix_Skel.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side glue code for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_SStructParCSRMatrix_IOR.h"
#include "bHYPRE_SStructParCSRMatrix.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructParCSRMatrix__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRMatrix__ctor(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRMatrix__ctor2(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRMatrix__dtor(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_SStructParCSRMatrix
impl_bHYPRE_SStructParCSRMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGraph graph,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Operator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Operator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructGraph(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructGraph(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructParCSRMatrix__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructParCSRMatrix(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructParCSRMatrix__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructParCSRMatrix(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Vector(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Vector(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetGraph(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_SStructGraph graph,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetComplex(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_Print(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_GetObject(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructParCSRMatrix_Destroy(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_Initialize(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_Assemble(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_Setup(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_Apply(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructParCSRMatrix_ApplyAdjoint(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Operator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Operator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructGraph(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructGraph(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixView(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_SStructParCSRMatrix__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructParCSRMatrix(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructParCSRMatrix__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructParCSRMatrix(void* bi, 
  sidl_BaseInterface* _ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Vector(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Vector(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in rarray[nentries] */ struct sidl_int__array* entries,
  /* in rarray[nentries] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1, 
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1, 
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp,
      _ex);
  sidl_int__array_deleteRef(index_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nentries] */ struct sidl_int__array* entries,
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
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1, 
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(entries_proxy,0);
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp,
      nvalues,
      _ex);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_AddToValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in rarray[nentries] */ struct sidl_int__array* entries,
  /* in rarray[nentries] */ struct sidl_double__array* values,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* index_proxy = sidl_int__array_ensure(index, 1, 
    sidl_column_major_order);
  int32_t* index_tmp = index_proxy->d_firstElement;
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1, 
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(index_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_AddToValues(
      self,
      part,
      index_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp,
      _ex);
  sidl_int__array_deleteRef(index_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nentries] */ struct sidl_int__array* entries,
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
  struct sidl_int__array* entries_proxy = sidl_int__array_ensure(entries, 1, 
    sidl_column_major_order);
  int32_t* entries_tmp = entries_proxy->d_firstElement;
  struct sidl_double__array* values_proxy = sidl_double__array_ensure(values, 1,
    sidl_column_major_order);
  double* values_tmp = values_proxy->d_firstElement;
  int32_t nentries = sidlLength(entries_proxy,0);
  int32_t nvalues = sidlLength(values_proxy,0);
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
      self,
      part,
      ilower_tmp,
      iupper_tmp,
      dim,
      var,
      nentries,
      entries_tmp,
      values_tmp,
      nvalues,
      _ex);
  sidl_int__array_deleteRef(ilower_proxy);
  sidl_int__array_deleteRef(iupper_proxy);
  sidl_int__array_deleteRef(entries_proxy);
  sidl_double__array_deleteRef(values_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
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
    impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues,
      _ex);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* value_proxy = sidl_int__array_ensure(value, 2, 
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
      self,
      name,
      value_proxy,
      _ex);
  sidl_int__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
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
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
      self,
      name,
      value_tmp,
      nvalues,
      _ex);
  sidl_double__array_deleteRef(value_proxy);
  return _return;
}

static int32_t
skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_double__array* value_proxy = sidl_double__array_ensure(value, 2, 
    sidl_column_major_order);
  _return =
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
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
bHYPRE_SStructParCSRMatrix__set_epv(struct bHYPRE_SStructParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructParCSRMatrix__ctor;
  epv->f__ctor2 = impl_bHYPRE_SStructParCSRMatrix__ctor2;
  epv->f__dtor = impl_bHYPRE_SStructParCSRMatrix__dtor;
  epv->f_SetGraph = impl_bHYPRE_SStructParCSRMatrix_SetGraph;
  epv->f_SetValues = skel_bHYPRE_SStructParCSRMatrix_SetValues;
  epv->f_SetBoxValues = skel_bHYPRE_SStructParCSRMatrix_SetBoxValues;
  epv->f_AddToValues = skel_bHYPRE_SStructParCSRMatrix_AddToValues;
  epv->f_AddToBoxValues = skel_bHYPRE_SStructParCSRMatrix_AddToBoxValues;
  epv->f_SetSymmetric = impl_bHYPRE_SStructParCSRMatrix_SetSymmetric;
  epv->f_SetNSSymmetric = impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric;
  epv->f_SetComplex = impl_bHYPRE_SStructParCSRMatrix_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructParCSRMatrix_Print;
  epv->f_GetObject = impl_bHYPRE_SStructParCSRMatrix_GetObject;
  epv->f_SetCommunicator = impl_bHYPRE_SStructParCSRMatrix_SetCommunicator;
  epv->f_Destroy = impl_bHYPRE_SStructParCSRMatrix_Destroy;
  epv->f_Initialize = impl_bHYPRE_SStructParCSRMatrix_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructParCSRMatrix_Assemble;
  epv->f_SetIntParameter = impl_bHYPRE_SStructParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = 
    impl_bHYPRE_SStructParCSRMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter = 
    skel_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue = impl_bHYPRE_SStructParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue;
  epv->f_Setup = impl_bHYPRE_SStructParCSRMatrix_Setup;
  epv->f_Apply = impl_bHYPRE_SStructParCSRMatrix_Apply;
  epv->f_ApplyAdjoint = impl_bHYPRE_SStructParCSRMatrix_ApplyAdjoint;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructParCSRMatrix__set_sepv(struct bHYPRE_SStructParCSRMatrix__sepv 
  *sepv)
{
  sepv->f_Create = impl_bHYPRE_SStructParCSRMatrix_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_SStructParCSRMatrix__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_SStructParCSRMatrix__load(&_throwaway_exception);
}
struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MPICommunicator(url, 
    ar, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(url, 
    ar, _ex);
}

struct bHYPRE_MatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MatrixVectorView(bi, _ex);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Operator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Operator(url, ar, _ex);
}

struct bHYPRE_Operator__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Operator(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Operator(bi, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(url, 
    ar, _ex);
}

struct bHYPRE_ProblemDefinition__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_ProblemDefinition(bi, 
    _ex);
}

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructGraph(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructGraph(url, ar, 
    _ex);
}

struct bHYPRE_SStructGraph__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructGraph(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructGraph(bi, _ex);
}

struct bHYPRE_SStructMatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return 
    impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixVectorView(url,
    ar, _ex);
}

struct bHYPRE_SStructMatrixVectorView__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixVectorView(
    bi, _ex);
}

struct bHYPRE_SStructMatrixView__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixView(url, 
    ar, _ex);
}

struct bHYPRE_SStructMatrixView__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixView(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixView(bi, 
    _ex);
}

struct bHYPRE_SStructParCSRMatrix__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructParCSRMatrix(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructParCSRMatrix(
    url, ar, _ex);
}

struct bHYPRE_SStructParCSRMatrix__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructParCSRMatrix(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructParCSRMatrix(bi, 
    _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Vector(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Vector(url, ar, _ex);
}

struct bHYPRE_Vector__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Vector(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Vector(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseInterface(url, ar, 
    _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_SStructParCSRMatrix_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_RuntimeException(url, ar,
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_SStructParCSRMatrix_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_RuntimeException(bi, _ex);
}

struct bHYPRE_SStructParCSRMatrix__data*
bHYPRE_SStructParCSRMatrix__get_data(bHYPRE_SStructParCSRMatrix self)
{
  return (struct bHYPRE_SStructParCSRMatrix__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructParCSRMatrix__set_data(
  bHYPRE_SStructParCSRMatrix self,
  struct bHYPRE_SStructParCSRMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
