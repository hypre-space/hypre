/*
 * File:          bHYPRE_IJParCSRVector_Skel.c
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 803
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_IJParCSRVector_IOR.h"
#include "bHYPRE_IJParCSRVector.h"
#include <stddef.h>

extern void
impl_bHYPRE_IJParCSRVector__ctor(
  bHYPRE_IJParCSRVector);

extern void
impl_bHYPRE_IJParCSRVector__dtor(
  bHYPRE_IJParCSRVector);

extern int32_t
impl_bHYPRE_IJParCSRVector_SetCommunicator(
  bHYPRE_IJParCSRVector,
  void*);

extern int32_t
impl_bHYPRE_IJParCSRVector_Initialize(
  bHYPRE_IJParCSRVector);

extern int32_t
impl_bHYPRE_IJParCSRVector_Assemble(
  bHYPRE_IJParCSRVector);

extern int32_t
impl_bHYPRE_IJParCSRVector_GetObject(
  bHYPRE_IJParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_bHYPRE_IJParCSRVector_SetLocalRange(
  bHYPRE_IJParCSRVector,
  int32_t,
  int32_t);

extern int32_t
impl_bHYPRE_IJParCSRVector_SetValues(
  bHYPRE_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_IJParCSRVector_AddToValues(
  bHYPRE_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_bHYPRE_IJParCSRVector_GetLocalRange(
  bHYPRE_IJParCSRVector,
  int32_t*,
  int32_t*);

extern int32_t
impl_bHYPRE_IJParCSRVector_GetValues(
  bHYPRE_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array**);

extern int32_t
impl_bHYPRE_IJParCSRVector_Print(
  bHYPRE_IJParCSRVector,
  const char*);

extern int32_t
impl_bHYPRE_IJParCSRVector_Read(
  bHYPRE_IJParCSRVector,
  const char*,
  void*);

extern int32_t
impl_bHYPRE_IJParCSRVector_Clear(
  bHYPRE_IJParCSRVector);

extern int32_t
impl_bHYPRE_IJParCSRVector_Copy(
  bHYPRE_IJParCSRVector,
  bHYPRE_Vector);

extern int32_t
impl_bHYPRE_IJParCSRVector_Clone(
  bHYPRE_IJParCSRVector,
  bHYPRE_Vector*);

extern int32_t
impl_bHYPRE_IJParCSRVector_Scale(
  bHYPRE_IJParCSRVector,
  double);

extern int32_t
impl_bHYPRE_IJParCSRVector_Dot(
  bHYPRE_IJParCSRVector,
  bHYPRE_Vector,
  double*);

extern int32_t
impl_bHYPRE_IJParCSRVector_Axpy(
  bHYPRE_IJParCSRVector,
  double,
  bHYPRE_Vector);

void
bHYPRE_IJParCSRVector__set_epv(struct bHYPRE_IJParCSRVector__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_IJParCSRVector__ctor;
  epv->f__dtor = impl_bHYPRE_IJParCSRVector__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_IJParCSRVector_SetCommunicator;
  epv->f_Initialize = impl_bHYPRE_IJParCSRVector_Initialize;
  epv->f_Assemble = impl_bHYPRE_IJParCSRVector_Assemble;
  epv->f_GetObject = impl_bHYPRE_IJParCSRVector_GetObject;
  epv->f_SetLocalRange = impl_bHYPRE_IJParCSRVector_SetLocalRange;
  epv->f_SetValues = impl_bHYPRE_IJParCSRVector_SetValues;
  epv->f_AddToValues = impl_bHYPRE_IJParCSRVector_AddToValues;
  epv->f_GetLocalRange = impl_bHYPRE_IJParCSRVector_GetLocalRange;
  epv->f_GetValues = impl_bHYPRE_IJParCSRVector_GetValues;
  epv->f_Print = impl_bHYPRE_IJParCSRVector_Print;
  epv->f_Read = impl_bHYPRE_IJParCSRVector_Read;
  epv->f_Clear = impl_bHYPRE_IJParCSRVector_Clear;
  epv->f_Copy = impl_bHYPRE_IJParCSRVector_Copy;
  epv->f_Clone = impl_bHYPRE_IJParCSRVector_Clone;
  epv->f_Scale = impl_bHYPRE_IJParCSRVector_Scale;
  epv->f_Dot = impl_bHYPRE_IJParCSRVector_Dot;
  epv->f_Axpy = impl_bHYPRE_IJParCSRVector_Axpy;
}

struct bHYPRE_IJParCSRVector__data*
bHYPRE_IJParCSRVector__get_data(bHYPRE_IJParCSRVector self)
{
  return (struct bHYPRE_IJParCSRVector__data*)(self ? self->d_data : NULL);
}

void bHYPRE_IJParCSRVector__set_data(
  bHYPRE_IJParCSRVector self,
  struct bHYPRE_IJParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
