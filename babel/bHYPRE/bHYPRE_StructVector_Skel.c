/*
 * File:          bHYPRE_StructVector_Skel.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1117
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_StructVector_IOR.h"
#include "bHYPRE_StructVector.h"
#include <stddef.h>

extern void
impl_bHYPRE_StructVector__ctor(
  bHYPRE_StructVector);

extern void
impl_bHYPRE_StructVector__dtor(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  bHYPRE_StructVector,
  void*);

extern int32_t
impl_bHYPRE_StructVector_Initialize(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_Assemble(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_GetObject(
  bHYPRE_StructVector,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_StructVector_SetGrid(
  bHYPRE_StructVector,
  bHYPRE_StructGrid);

extern int32_t
impl_bHYPRE_StructVector_SetStencil(
  bHYPRE_StructVector,
  bHYPRE_StructStencil);

extern int32_t
impl_bHYPRE_StructVector_SetValue(
  bHYPRE_StructVector,
  struct SIDL_int__array*,
  double);

extern int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  bHYPRE_StructVector,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_StructVector_Clear(
  bHYPRE_StructVector);

extern int32_t
impl_bHYPRE_StructVector_Copy(
  bHYPRE_StructVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_StructVector_Clone(
  bHYPRE_StructVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_StructVector_Scale(
  bHYPRE_StructVector,
  double);

extern int32_t
impl_bHYPRE_StructVector_Dot(
  bHYPRE_StructVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_StructVector_Axpy(
  bHYPRE_StructVector,
  double,
  bHYPRE_Vector);

void
bHYPRE_StructVector__set_epv(struct bHYPRE_StructVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructVector__ctor;
  epv->f__dtor = impl_bHYPRE_StructVector__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_StructVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_StructVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_StructVector_GetObject;
  epv->f_SetGrid = impl_bHYPRE_StructVector_SetGrid;
  epv->f_SetStencil = impl_bHYPRE_StructVector_SetStencil;
  epv->f_SetValue = impl_bHYPRE_StructVector_SetValue;
  epv->f_SetBoxValues = impl_bHYPRE_StructVector_SetBoxValues;
  epv->f_Clear = impl_bHYPRE_StructVector_Clear;
  epv->f_Copy = impl_bHYPRE_StructVector_Copy;
  epv->f_Clone = impl_bHYPRE_StructVector_Clone;
  epv->f_Scale = impl_bHYPRE_StructVector_Scale;
  epv->f_Dot = impl_bHYPRE_StructVector_Dot;
  epv->f_Axpy = impl_bHYPRE_StructVector_Axpy;
}

struct bHYPRE_StructVector__data*
bHYPRE_StructVector__get_data(bHYPRE_StructVector self)
{
  return (struct bHYPRE_StructVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructVector__set_data(
  bHYPRE_StructVector self,
  struct bHYPRE_StructVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
