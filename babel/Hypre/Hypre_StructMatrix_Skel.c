/*
 * File:          Hypre_StructMatrix_Skel.c
 * Symbol:        Hypre.StructMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:33 PDT
 * Description:   Server-side glue code for Hypre.StructMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructMatrix_IOR.h"
#include "Hypre_StructMatrix.h"
#include <stddef.h>

extern void
impl_Hypre_StructMatrix__ctor(
  Hypre_StructMatrix);

extern void
impl_Hypre_StructMatrix__dtor(
  Hypre_StructMatrix);

extern int32_t
impl_Hypre_StructMatrix_Apply(
  Hypre_StructMatrix,
  Hypre_Vector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_StructMatrix_Assemble(
  Hypre_StructMatrix);

extern int32_t
impl_Hypre_StructMatrix_GetDoubleValue(
  Hypre_StructMatrix,
  const char*,
  double*);

extern int32_t
impl_Hypre_StructMatrix_GetIntValue(
  Hypre_StructMatrix,
  const char*,
  int32_t*);

extern int32_t
impl_Hypre_StructMatrix_GetObject(
  Hypre_StructMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_StructMatrix_Initialize(
  Hypre_StructMatrix);

extern int32_t
impl_Hypre_StructMatrix_SetBoxValues(
  Hypre_StructMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructMatrix_SetCommunicator(
  Hypre_StructMatrix,
  void*);

extern int32_t
impl_Hypre_StructMatrix_SetDoubleArrayParameter(
  Hypre_StructMatrix,
  const char*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructMatrix_SetDoubleParameter(
  Hypre_StructMatrix,
  const char*,
  double);

extern int32_t
impl_Hypre_StructMatrix_SetGrid(
  Hypre_StructMatrix,
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructMatrix_SetIntArrayParameter(
  Hypre_StructMatrix,
  const char*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructMatrix_SetIntParameter(
  Hypre_StructMatrix,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_StructMatrix_SetNumGhost(
  Hypre_StructMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructMatrix_SetStencil(
  Hypre_StructMatrix,
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructMatrix_SetStringParameter(
  Hypre_StructMatrix,
  const char*,
  const char*);

extern int32_t
impl_Hypre_StructMatrix_SetSymmetric(
  Hypre_StructMatrix,
  int32_t);

extern int32_t
impl_Hypre_StructMatrix_SetValues(
  Hypre_StructMatrix,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructMatrix_Setup(
  Hypre_StructMatrix);

void
Hypre_StructMatrix__set_epv(struct Hypre_StructMatrix__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructMatrix__ctor;
  epv->f__dtor = impl_Hypre_StructMatrix__dtor;
  epv->f_Setup = impl_Hypre_StructMatrix_Setup;
  epv->f_SetIntArrayParameter = impl_Hypre_StructMatrix_SetIntArrayParameter;
  epv->f_SetIntParameter = impl_Hypre_StructMatrix_SetIntParameter;
  epv->f_SetStencil = impl_Hypre_StructMatrix_SetStencil;
  epv->f_GetIntValue = impl_Hypre_StructMatrix_GetIntValue;
  epv->f_SetCommunicator = impl_Hypre_StructMatrix_SetCommunicator;
  epv->f_SetDoubleParameter = impl_Hypre_StructMatrix_SetDoubleParameter;
  epv->f_SetStringParameter = impl_Hypre_StructMatrix_SetStringParameter;
  epv->f_SetSymmetric = impl_Hypre_StructMatrix_SetSymmetric;
  epv->f_GetObject = impl_Hypre_StructMatrix_GetObject;
  epv->f_Assemble = impl_Hypre_StructMatrix_Assemble;
  epv->f_Initialize = impl_Hypre_StructMatrix_Initialize;
  epv->f_Apply = impl_Hypre_StructMatrix_Apply;
  epv->f_SetNumGhost = impl_Hypre_StructMatrix_SetNumGhost;
  epv->f_SetBoxValues = impl_Hypre_StructMatrix_SetBoxValues;
  epv->f_GetDoubleValue = impl_Hypre_StructMatrix_GetDoubleValue;
  epv->f_SetGrid = impl_Hypre_StructMatrix_SetGrid;
  epv->f_SetValues = impl_Hypre_StructMatrix_SetValues;
  epv->f_SetDoubleArrayParameter = 
    impl_Hypre_StructMatrix_SetDoubleArrayParameter;
}

struct Hypre_StructMatrix__data*
Hypre_StructMatrix__get_data(Hypre_StructMatrix self)
{
  return (struct Hypre_StructMatrix__data*)(self ? self->d_data : NULL);
}

void Hypre_StructMatrix__set_data(
  Hypre_StructMatrix self,
  struct Hypre_StructMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
