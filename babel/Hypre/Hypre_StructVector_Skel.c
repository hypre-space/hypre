/*
 * File:          Hypre_StructVector_Skel.c
 * Symbol:        Hypre.StructVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
 * Description:   Server-side glue code for Hypre.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructVector_IOR.h"
#include "Hypre_StructVector.h"
#include <stddef.h>

extern void
impl_Hypre_StructVector__ctor(
  Hypre_StructVector);

extern void
impl_Hypre_StructVector__dtor(
  Hypre_StructVector);

extern int32_t
impl_Hypre_StructVector_Assemble(
  Hypre_StructVector);

extern int32_t
impl_Hypre_StructVector_Axpy(
  Hypre_StructVector,
  double,
  Hypre_Vector);

extern int32_t
impl_Hypre_StructVector_Clear(
  Hypre_StructVector);

extern int32_t
impl_Hypre_StructVector_Clone(
  Hypre_StructVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_StructVector_Copy(
  Hypre_StructVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_StructVector_Dot(
  Hypre_StructVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_StructVector_GetObject(
  Hypre_StructVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_StructVector_Initialize(
  Hypre_StructVector);

extern int32_t
impl_Hypre_StructVector_Scale(
  Hypre_StructVector,
  double);

extern int32_t
impl_Hypre_StructVector_SetBoxValues(
  Hypre_StructVector,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructVector_SetCommunicator(
  Hypre_StructVector,
  void*);

extern int32_t
impl_Hypre_StructVector_SetGrid(
  Hypre_StructVector,
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructVector_SetStencil(
  Hypre_StructVector,
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructVector_SetValue(
  Hypre_StructVector,
  struct SIDL_int__array*,
  double);

void
Hypre_StructVector__set_epv(struct Hypre_StructVector__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructVector__ctor;
  epv->f__dtor = impl_Hypre_StructVector__dtor;
  epv->f_Axpy = impl_Hypre_StructVector_Axpy;
  epv->f_Initialize = impl_Hypre_StructVector_Initialize;
  epv->f_SetCommunicator = impl_Hypre_StructVector_SetCommunicator;
  epv->f_SetStencil = impl_Hypre_StructVector_SetStencil;
  epv->f_SetValue = impl_Hypre_StructVector_SetValue;
  epv->f_Clone = impl_Hypre_StructVector_Clone;
  epv->f_Clear = impl_Hypre_StructVector_Clear;
  epv->f_Assemble = impl_Hypre_StructVector_Assemble;
  epv->f_SetBoxValues = impl_Hypre_StructVector_SetBoxValues;
  epv->f_Scale = impl_Hypre_StructVector_Scale;
  epv->f_Dot = impl_Hypre_StructVector_Dot;
  epv->f_GetObject = impl_Hypre_StructVector_GetObject;
  epv->f_Copy = impl_Hypre_StructVector_Copy;
  epv->f_SetGrid = impl_Hypre_StructVector_SetGrid;
}

struct Hypre_StructVector__data*
Hypre_StructVector__get_data(Hypre_StructVector self)
{
  return (struct Hypre_StructVector__data*)(self ? self->d_data : NULL);
}

void Hypre_StructVector__set_data(
  Hypre_StructVector self,
  struct Hypre_StructVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
