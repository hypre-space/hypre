/*
 * File:          bHYPRE_SStructVector_Skel.c
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1062
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructVector_IOR.h"
#include "bHYPRE_SStructVector.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructVector__ctor(
  bHYPRE_SStructVector);

extern void
impl_bHYPRE_SStructVector__dtor(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_SetCommunicator(
  bHYPRE_SStructVector,
  void*);

extern int32_t
impl_bHYPRE_SStructVector_Initialize(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Assemble(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_GetObject(
  bHYPRE_SStructVector,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_SStructVector_SetGrid(
  bHYPRE_SStructVector,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructVector_SetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_SetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_AddToBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_SStructVector_Gather(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_GetValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  double*);

extern int32_t
impl_bHYPRE_SStructVector_GetBoxValues(
  bHYPRE_SStructVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_double__array**);

extern int32_t
impl_bHYPRE_SStructVector_SetComplex(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Print(
  bHYPRE_SStructVector,
  const char*,
  int32_t);

extern int32_t
impl_bHYPRE_SStructVector_Clear(
  bHYPRE_SStructVector);

extern int32_t
impl_bHYPRE_SStructVector_Copy(
  bHYPRE_SStructVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_SStructVector_Clone(
  bHYPRE_SStructVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_SStructVector_Scale(
  bHYPRE_SStructVector,
  double);

extern int32_t
impl_bHYPRE_SStructVector_Dot(
  bHYPRE_SStructVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_SStructVector_Axpy(
  bHYPRE_SStructVector,
  double,
  bHYPRE_Vector);

void
bHYPRE_SStructVector__set_epv(struct bHYPRE_SStructVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructVector__ctor;
  epv->f__dtor = impl_bHYPRE_SStructVector__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_SStructVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_SStructVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_SStructVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_SStructVector_GetObject;
  epv->f_SetGrid = impl_bHYPRE_SStructVector_SetGrid;
  epv->f_SetValues = impl_bHYPRE_SStructVector_SetValues;
  epv->f_SetBoxValues = impl_bHYPRE_SStructVector_SetBoxValues;
  epv->f_AddToValues = impl_bHYPRE_SStructVector_AddToValues;
  epv->f_AddToBoxValues = impl_bHYPRE_SStructVector_AddToBoxValues;
  epv->f_Gather = impl_bHYPRE_SStructVector_Gather;
  epv->f_GetValues = impl_bHYPRE_SStructVector_GetValues;
  epv->f_GetBoxValues = impl_bHYPRE_SStructVector_GetBoxValues;
  epv->f_SetComplex = impl_bHYPRE_SStructVector_SetComplex;
  epv->f_Print = impl_bHYPRE_SStructVector_Print;
  epv->f_Clear = impl_bHYPRE_SStructVector_Clear;
  epv->f_Copy = impl_bHYPRE_SStructVector_Copy;
  epv->f_Clone = impl_bHYPRE_SStructVector_Clone;
  epv->f_Scale = impl_bHYPRE_SStructVector_Scale;
  epv->f_Dot = impl_bHYPRE_SStructVector_Dot;
  epv->f_Axpy = impl_bHYPRE_SStructVector_Axpy;
}

struct bHYPRE_SStructVector__data*
bHYPRE_SStructVector__get_data(bHYPRE_SStructVector self)
{
  return (struct bHYPRE_SStructVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructVector__set_data(
  bHYPRE_SStructVector self,
  struct bHYPRE_SStructVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
