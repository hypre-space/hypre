/*
 * File:          Hypre_SStructParCSRVector_Skel.c
 * Symbol:        Hypre.SStructParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 847
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_SStructParCSRVector_IOR.h"
#include "Hypre_SStructParCSRVector.h"
#include <stddef.h>

extern void
impl_Hypre_SStructParCSRVector__ctor(
  Hypre_SStructParCSRVector);

extern void
impl_Hypre_SStructParCSRVector__dtor(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_SetCommunicator(
  Hypre_SStructParCSRVector,
  void*);

extern int32_t
impl_Hypre_SStructParCSRVector_Initialize(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_Assemble(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_GetObject(
  Hypre_SStructParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_SStructParCSRVector_SetGrid(
  Hypre_SStructParCSRVector,
  Hypre_SStructGrid);

extern int32_t
impl_Hypre_SStructParCSRVector_SetValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_SetBoxValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_AddToValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_AddToBoxValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_SStructParCSRVector_Gather(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_GetValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_Hypre_SStructParCSRVector_GetBoxValues(
  Hypre_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_SStructParCSRVector_SetComplex(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_Print(
  Hypre_SStructParCSRVector,
  const char*,
  int32_t);

extern int32_t
impl_Hypre_SStructParCSRVector_Clear(
  Hypre_SStructParCSRVector);

extern int32_t
impl_Hypre_SStructParCSRVector_Copy(
  Hypre_SStructParCSRVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_SStructParCSRVector_Clone(
  Hypre_SStructParCSRVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_SStructParCSRVector_Scale(
  Hypre_SStructParCSRVector,
  double);

extern int32_t
impl_Hypre_SStructParCSRVector_Dot(
  Hypre_SStructParCSRVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_SStructParCSRVector_Axpy(
  Hypre_SStructParCSRVector,
  double,
  Hypre_Vector);

void
Hypre_SStructParCSRVector__set_epv(struct Hypre_SStructParCSRVector__epv *epv)
{
  epv->f__ctor = impl_Hypre_SStructParCSRVector__ctor;
  epv->f__dtor = impl_Hypre_SStructParCSRVector__dtor;
  epv->f_SetCommunicator = impl_Hypre_SStructParCSRVector_SetCommunicator;
  epv->f_Initialize = impl_Hypre_SStructParCSRVector_Initialize;
  epv->f_Assemble = impl_Hypre_SStructParCSRVector_Assemble;
  epv->f_GetObject = impl_Hypre_SStructParCSRVector_GetObject;
  epv->f_SetGrid = impl_Hypre_SStructParCSRVector_SetGrid;
  epv->f_SetValues = impl_Hypre_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues = impl_Hypre_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues = impl_Hypre_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues = impl_Hypre_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather = impl_Hypre_SStructParCSRVector_Gather;
  epv->f_GetValues = impl_Hypre_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues = impl_Hypre_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex = impl_Hypre_SStructParCSRVector_SetComplex;
  epv->f_Print = impl_Hypre_SStructParCSRVector_Print;
  epv->f_Clear = impl_Hypre_SStructParCSRVector_Clear;
  epv->f_Copy = impl_Hypre_SStructParCSRVector_Copy;
  epv->f_Clone = impl_Hypre_SStructParCSRVector_Clone;
  epv->f_Scale = impl_Hypre_SStructParCSRVector_Scale;
  epv->f_Dot = impl_Hypre_SStructParCSRVector_Dot;
  epv->f_Axpy = impl_Hypre_SStructParCSRVector_Axpy;
}

struct Hypre_SStructParCSRVector__data*
Hypre_SStructParCSRVector__get_data(Hypre_SStructParCSRVector self)
{
  return (struct Hypre_SStructParCSRVector__data*)(self ? self->d_data : NULL);
}

void Hypre_SStructParCSRVector__set_data(
  Hypre_SStructParCSRVector self,
  struct Hypre_SStructParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
