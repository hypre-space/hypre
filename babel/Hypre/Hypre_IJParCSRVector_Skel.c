/*
 * File:          Hypre_IJParCSRVector_Skel.c
 * Symbol:        Hypre.IJParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 825
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_IJParCSRVector_IOR.h"
#include "Hypre_IJParCSRVector.h"
#include <stddef.h>

extern void
impl_Hypre_IJParCSRVector__ctor(
  Hypre_IJParCSRVector);

extern void
impl_Hypre_IJParCSRVector__dtor(
  Hypre_IJParCSRVector);

extern int32_t
impl_Hypre_IJParCSRVector_SetCommunicator(
  Hypre_IJParCSRVector,
  void*);

extern int32_t
impl_Hypre_IJParCSRVector_Initialize(
  Hypre_IJParCSRVector);

extern int32_t
impl_Hypre_IJParCSRVector_Assemble(
  Hypre_IJParCSRVector);

extern int32_t
impl_Hypre_IJParCSRVector_GetObject(
  Hypre_IJParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_IJParCSRVector_SetLocalRange(
  Hypre_IJParCSRVector,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_IJParCSRVector_SetValues(
  Hypre_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRVector_AddToValues(
  Hypre_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_IJParCSRVector_GetLocalRange(
  Hypre_IJParCSRVector,
  int32_t*,
  int32_t*);

extern int32_t
impl_Hypre_IJParCSRVector_GetValues(
  Hypre_IJParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_IJParCSRVector_Print(
  Hypre_IJParCSRVector,
  const char*);

extern int32_t
impl_Hypre_IJParCSRVector_Read(
  Hypre_IJParCSRVector,
  const char*,
  void*);

extern int32_t
impl_Hypre_IJParCSRVector_Clear(
  Hypre_IJParCSRVector);

extern int32_t
impl_Hypre_IJParCSRVector_Copy(
  Hypre_IJParCSRVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_IJParCSRVector_Clone(
  Hypre_IJParCSRVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_IJParCSRVector_Scale(
  Hypre_IJParCSRVector,
  double);

extern int32_t
impl_Hypre_IJParCSRVector_Dot(
  Hypre_IJParCSRVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_IJParCSRVector_Axpy(
  Hypre_IJParCSRVector,
  double,
  Hypre_Vector);

void
Hypre_IJParCSRVector__set_epv(struct Hypre_IJParCSRVector__epv *epv)
{
  epv->f__ctor = impl_Hypre_IJParCSRVector__ctor;
  epv->f__dtor = impl_Hypre_IJParCSRVector__dtor;
  epv->f_SetCommunicator = impl_Hypre_IJParCSRVector_SetCommunicator;
  epv->f_Initialize = impl_Hypre_IJParCSRVector_Initialize;
  epv->f_Assemble = impl_Hypre_IJParCSRVector_Assemble;
  epv->f_GetObject = impl_Hypre_IJParCSRVector_GetObject;
  epv->f_SetLocalRange = impl_Hypre_IJParCSRVector_SetLocalRange;
  epv->f_SetValues = impl_Hypre_IJParCSRVector_SetValues;
  epv->f_AddToValues = impl_Hypre_IJParCSRVector_AddToValues;
  epv->f_GetLocalRange = impl_Hypre_IJParCSRVector_GetLocalRange;
  epv->f_GetValues = impl_Hypre_IJParCSRVector_GetValues;
  epv->f_Print = impl_Hypre_IJParCSRVector_Print;
  epv->f_Read = impl_Hypre_IJParCSRVector_Read;
  epv->f_Clear = impl_Hypre_IJParCSRVector_Clear;
  epv->f_Copy = impl_Hypre_IJParCSRVector_Copy;
  epv->f_Clone = impl_Hypre_IJParCSRVector_Clone;
  epv->f_Scale = impl_Hypre_IJParCSRVector_Scale;
  epv->f_Dot = impl_Hypre_IJParCSRVector_Dot;
  epv->f_Axpy = impl_Hypre_IJParCSRVector_Axpy;
}

struct Hypre_IJParCSRVector__data*
Hypre_IJParCSRVector__get_data(Hypre_IJParCSRVector self)
{
  return (struct Hypre_IJParCSRVector__data*)(self ? self->d_data : NULL);
}

void Hypre_IJParCSRVector__set_data(
  Hypre_IJParCSRVector self,
  struct Hypre_IJParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
