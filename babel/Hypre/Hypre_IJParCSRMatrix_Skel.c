/*
 * File:          Hypre_IJParCSRMatrix_Skel.c
 * Symbol:        Hypre.IJParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 799
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_IJParCSRMatrix_IOR.h"
#include "Hypre_IJParCSRMatrix.h"
#include <stddef.h>

extern void
impl_Hypre_IJParCSRMatrix__ctor(
  Hypre_IJParCSRMatrix);

extern void
impl_Hypre_IJParCSRMatrix__dtor(
  Hypre_IJParCSRMatrix);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetDiagOffdSizes(
  Hypre_IJParCSRMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetCommunicator(
  Hypre_IJParCSRMatrix,
  void*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetIntParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetDoubleParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetStringParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  const char*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetIntArrayParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetDoubleArrayParameter(
  Hypre_IJParCSRMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetIntValue(
  Hypre_IJParCSRMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetDoubleValue(
  Hypre_IJParCSRMatrix,
  const char*,
  double*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Setup(
  Hypre_IJParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector);

extern int32_t
impl_Hypre_IJParCSRMatrix_Apply(
  Hypre_IJParCSRMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Initialize(
  Hypre_IJParCSRMatrix);

extern int32_t
impl_Hypre_IJParCSRMatrix_Assemble(
  Hypre_IJParCSRMatrix);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetObject(
  Hypre_IJParCSRMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetLocalRange(
  Hypre_IJParCSRMatrix,
  int32_t,
  int32_t,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetValues(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_AddToValues(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetLocalRange(
  Hypre_IJParCSRMatrix,
  int32_t*,
  int32_t*,
  int32_t*,
  int32_t*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetRowCounts(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array**);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetValues(
  Hypre_IJParCSRMatrix,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_IJParCSRMatrix_SetRowSizes(
  Hypre_IJParCSRMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Print(
  Hypre_IJParCSRMatrix,
  const char*);

extern int32_t
impl_Hypre_IJParCSRMatrix_Read(
  Hypre_IJParCSRMatrix,
  const char*,
  void*);

extern int32_t
impl_Hypre_IJParCSRMatrix_GetRow(
  Hypre_IJParCSRMatrix,
  int32_t,
  int32_t*,
  struct SIDL_int__array**,
  struct SIDL_double__array**);

void
Hypre_IJParCSRMatrix__set_epv(struct Hypre_IJParCSRMatrix__epv *epv)
{
  epv->f__ctor = impl_Hypre_IJParCSRMatrix__ctor;
  epv->f__dtor = impl_Hypre_IJParCSRMatrix__dtor;
  epv->f_SetDiagOffdSizes = impl_Hypre_IJParCSRMatrix_SetDiagOffdSizes;
  epv->f_SetCommunicator = impl_Hypre_IJParCSRMatrix_SetCommunicator;
  epv->f_SetIntParameter = impl_Hypre_IJParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter = impl_Hypre_IJParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_IJParCSRMatrix_SetStringParameter;
  epv->f_SetIntArrayParameter = impl_Hypre_IJParCSRMatrix_SetIntArrayParameter;
  epv->f_SetDoubleArrayParameter = 
    impl_Hypre_IJParCSRMatrix_SetDoubleArrayParameter;
  epv->f_GetIntValue = impl_Hypre_IJParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue = impl_Hypre_IJParCSRMatrix_GetDoubleValue;
  epv->f_Setup = impl_Hypre_IJParCSRMatrix_Setup;
  epv->f_Apply = impl_Hypre_IJParCSRMatrix_Apply;
  epv->f_Initialize = impl_Hypre_IJParCSRMatrix_Initialize;
  epv->f_Assemble = impl_Hypre_IJParCSRMatrix_Assemble;
  epv->f_GetObject = impl_Hypre_IJParCSRMatrix_GetObject;
  epv->f_SetLocalRange = impl_Hypre_IJParCSRMatrix_SetLocalRange;
  epv->f_SetValues = impl_Hypre_IJParCSRMatrix_SetValues;
  epv->f_AddToValues = impl_Hypre_IJParCSRMatrix_AddToValues;
  epv->f_GetLocalRange = impl_Hypre_IJParCSRMatrix_GetLocalRange;
  epv->f_GetRowCounts = impl_Hypre_IJParCSRMatrix_GetRowCounts;
  epv->f_GetValues = impl_Hypre_IJParCSRMatrix_GetValues;
  epv->f_SetRowSizes = impl_Hypre_IJParCSRMatrix_SetRowSizes;
  epv->f_Print = impl_Hypre_IJParCSRMatrix_Print;
  epv->f_Read = impl_Hypre_IJParCSRMatrix_Read;
  epv->f_GetRow = impl_Hypre_IJParCSRMatrix_GetRow;
}

struct Hypre_IJParCSRMatrix__data*
Hypre_IJParCSRMatrix__get_data(Hypre_IJParCSRMatrix self)
{
  return (struct Hypre_IJParCSRMatrix__data*)(self ? self->d_data : NULL);
}

void Hypre_IJParCSRMatrix__set_data(
  Hypre_IJParCSRMatrix self,
  struct Hypre_IJParCSRMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
