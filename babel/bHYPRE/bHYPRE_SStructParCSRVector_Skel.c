/*
 * File:          bHYPRE_SStructParCSRVector_Skel.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 825
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructParCSRVector_IOR.h"
#include "bHYPRE_SStructParCSRVector.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructParCSRVector__ctor(
  bHYPRE_SStructParCSRVector);

extern void
impl_bHYPRE_SStructParCSRVector__dtor(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  bHYPRE_SStructParCSRVector,
  void*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  bHYPRE_SStructParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  bHYPRE_SStructParCSRVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  bHYPRE_SStructParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  bHYPRE_SStructParCSRVector,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  bHYPRE_SStructParCSRVector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  bHYPRE_SStructParCSRVector,
  double);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  bHYPRE_SStructParCSRVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  bHYPRE_SStructParCSRVector,
  double,
  bHYPRE_Vector);

void
bHYPRE_SStructParCSRVector__set_epv(struct bHYPRE_SStructParCSRVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructParCSRVector__ctor;
  epv->f__dtor = impl_bHYPRE_SStructParCSRVector__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructParCSRVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_SStructParCSRVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructParCSRVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructParCSRVector_GetObject;
  epv->f_SetGrid = impl_bHYPRE_SStructParCSRVector_SetGrid;
  epv->f_SetValues = impl_bHYPRE_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues = impl_bHYPRE_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues = impl_bHYPRE_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues = impl_bHYPRE_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather = impl_bHYPRE_SStructParCSRVector_Gather;
  epv->f_GetValues = impl_bHYPRE_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues = impl_bHYPRE_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex = impl_bHYPRE_SStructParCSRVector_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructParCSRVector_Print;
  epv->f_Clear = impl_bHYPRE_SStructParCSRVector_Clear;
  epv->f_Copy = impl_bHYPRE_SStructParCSRVector_Copy;
  epv->f_Clone = impl_bHYPRE_SStructParCSRVector_Clone;
  epv->f_Scale = impl_bHYPRE_SStructParCSRVector_Scale;
  epv->f_Dot = impl_bHYPRE_SStructParCSRVector_Dot;
  epv->f_Axpy = impl_bHYPRE_SStructParCSRVector_Axpy;
}

struct bHYPRE_SStructParCSRVector__data*
bHYPRE_SStructParCSRVector__get_data(bHYPRE_SStructParCSRVector self)
{
  return (struct bHYPRE_SStructParCSRVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructParCSRVector__set_data(
  bHYPRE_SStructParCSRVector self,
  struct bHYPRE_SStructParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
