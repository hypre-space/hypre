/*
 * File:          Hypre_SStructVector_Skel.c
 * Symbol:        Hypre.SStructVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1084
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_SStructVector_IOR.h"
#include "Hypre_SStructVector.h"
#include <stddef.h>

extern void
impl_Hypre_SStructVector__ctor(
  Hypre_SStructVector);

extern void
impl_Hypre_SStructVector__dtor(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_SetCommunicator(
  Hypre_SStructVector,
  void*);

extern int32_t
impl_Hypre_SStructVector_Initialize(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_Assemble(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_GetObject(
  Hypre_SStructVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_SStructVector_SetGrid(
  Hypre_SStructVector,
  Hypre_SStructGrid);

extern int32_t
impl_Hypre_SStructVector_SetValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_SetBoxValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_AddToValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_AddToBoxValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructVector_Gather(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_GetValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_Hypre_SStructVector_GetBoxValues(
  Hypre_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_SStructVector_SetComplex(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_Print(
  Hypre_SStructVector,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_SStructVector_Clear(
  Hypre_SStructVector);

extern int32_t
impl_Hypre_SStructVector_Copy(
  Hypre_SStructVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_SStructVector_Clone(
  Hypre_SStructVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_SStructVector_Scale(
  Hypre_SStructVector,
  double);

extern int32_t
impl_Hypre_SStructVector_Dot(
  Hypre_SStructVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_SStructVector_Axpy(
  Hypre_SStructVector,
  double,
  Hypre_Vector);

void
Hypre_SStructVector__set_epv(struct Hypre_SStructVector__epv *epv)
{
  epv->f__ctor = impl_Hypre_SStructVector__ctor;
  epv->f__dtor = impl_Hypre_SStructVector__dtor;
  epv->f_SetCommunicator = impl_Hypre_SStructVector_SetCommunicator;
  epv->f_Initialize = impl_Hypre_SStructVector_Initialize;
  epv->f_Assemble = impl_Hypre_SStructVector_Assemble;
  epv->f_GetObject = impl_Hypre_SStructVector_GetObject;
  epv->f_SetGrid = impl_Hypre_SStructVector_SetGrid;
  epv->f_SetValues = impl_Hypre_SStructVector_SetValues;
  epv->f_SetBoxValues = impl_Hypre_SStructVector_SetBoxValues;
  epv->f_AddToValues = impl_Hypre_SStructVector_AddToValues;
  epv->f_AddToBoxValues = impl_Hypre_SStructVector_AddToBoxValues;
  epv->f_Gather = impl_Hypre_SStructVector_Gather;
  epv->f_GetValues = impl_Hypre_SStructVector_GetValues;
  epv->f_GetBoxValues = impl_Hypre_SStructVector_GetBoxValues;
  epv->f_SetComplex = impl_Hypre_SStructVector_SetComplex;
  epv->f_Print = impl_Hypre_SStructVector_Print;
  epv->f_Clear = impl_Hypre_SStructVector_Clear;
  epv->f_Copy = impl_Hypre_SStructVector_Copy;
  epv->f_Clone = impl_Hypre_SStructVector_Clone;
  epv->f_Scale = impl_Hypre_SStructVector_Scale;
  epv->f_Dot = impl_Hypre_SStructVector_Dot;
  epv->f_Axpy = impl_Hypre_SStructVector_Axpy;
}

struct Hypre_SStructVector__data*
Hypre_SStructVector__get_data(Hypre_SStructVector self)
{
  return (struct Hypre_SStructVector__data*)(self ? self->d_data : NULL);
}

void Hypre_SStructVector__set_data(
  Hypre_SStructVector self,
  struct Hypre_SStructVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
