/*
 * File:          Hypre_ParCSRVector_Skel.c
 * Symbol:        Hypre.ParCSRVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:54 PDT
 * Description:   Server-side glue code for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_ParCSRVector_IOR.h"
#include "Hypre_ParCSRVector.h"
#include <stddef.h>

extern void
impl_Hypre_ParCSRVector__ctor(
  Hypre_ParCSRVector);

extern void
impl_Hypre_ParCSRVector__dtor(
  Hypre_ParCSRVector);

extern int32_t
impl_Hypre_ParCSRVector_AddToLocalComponentsInBlock(
  Hypre_ParCSRVector,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_AddToValues(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_AddtoLocalComponents(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_Assemble(
  Hypre_ParCSRVector);

extern int32_t
impl_Hypre_ParCSRVector_Axpy(
  Hypre_ParCSRVector,
  double,
  Hypre_Vector);

extern int32_t
impl_Hypre_ParCSRVector_Clear(
  Hypre_ParCSRVector);

extern int32_t
impl_Hypre_ParCSRVector_Clone(
  Hypre_ParCSRVector,
  Hypre_Vector*);

extern int32_t
impl_Hypre_ParCSRVector_Copy(
  Hypre_ParCSRVector,
  Hypre_Vector);

extern int32_t
impl_Hypre_ParCSRVector_Create(
  Hypre_ParCSRVector,
  void*,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_ParCSRVector_Dot(
  Hypre_ParCSRVector,
  Hypre_Vector,
  double*);

extern int32_t
impl_Hypre_ParCSRVector_GetObject(
  Hypre_ParCSRVector,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_ParCSRVector_GetRow(
  Hypre_ParCSRVector,
  int32_t,
  int32_t*,
  struct SIDL_int__array**,
  struct SIDL_double__array**);

extern int32_t
impl_Hypre_ParCSRVector_Initialize(
  Hypre_ParCSRVector);

extern int32_t
impl_Hypre_ParCSRVector_Print(
  Hypre_ParCSRVector,
  const char*);

extern int32_t
impl_Hypre_ParCSRVector_Read(
  Hypre_ParCSRVector,
  const char*,
  void*);

extern int32_t
impl_Hypre_ParCSRVector_Scale(
  Hypre_ParCSRVector,
  double);

extern int32_t
impl_Hypre_ParCSRVector_SetCommunicator(
  Hypre_ParCSRVector,
  void*);

extern int32_t
impl_Hypre_ParCSRVector_SetGlobalSize(
  Hypre_ParCSRVector,
  int32_t);

extern int32_t
impl_Hypre_ParCSRVector_SetLocalComponents(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_SetLocalComponentsInBlock(
  Hypre_ParCSRVector,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_ParCSRVector_SetPartitioning(
  Hypre_ParCSRVector,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_ParCSRVector_SetValues(
  Hypre_ParCSRVector,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

void
Hypre_ParCSRVector__set_epv(struct Hypre_ParCSRVector__epv *epv)
{
  epv->f__ctor = impl_Hypre_ParCSRVector__ctor;
  epv->f__dtor = impl_Hypre_ParCSRVector__dtor;
  epv->f_SetLocalComponents = impl_Hypre_ParCSRVector_SetLocalComponents;
  epv->f_AddToValues = impl_Hypre_ParCSRVector_AddToValues;
  epv->f_Clone = impl_Hypre_ParCSRVector_Clone;
  epv->f_Create = impl_Hypre_ParCSRVector_Create;
  epv->f_SetPartitioning = impl_Hypre_ParCSRVector_SetPartitioning;
  epv->f_SetCommunicator = impl_Hypre_ParCSRVector_SetCommunicator;
  epv->f_Read = impl_Hypre_ParCSRVector_Read;
  epv->f_Copy = impl_Hypre_ParCSRVector_Copy;
  epv->f_GetObject = impl_Hypre_ParCSRVector_GetObject;
  epv->f_Assemble = impl_Hypre_ParCSRVector_Assemble;
  epv->f_Initialize = impl_Hypre_ParCSRVector_Initialize;
  epv->f_Clear = impl_Hypre_ParCSRVector_Clear;
  epv->f_Print = impl_Hypre_ParCSRVector_Print;
  epv->f_Scale = impl_Hypre_ParCSRVector_Scale;
  epv->f_Dot = impl_Hypre_ParCSRVector_Dot;
  epv->f_SetLocalComponentsInBlock = 
    impl_Hypre_ParCSRVector_SetLocalComponentsInBlock;
  epv->f_SetGlobalSize = impl_Hypre_ParCSRVector_SetGlobalSize;
  epv->f_AddtoLocalComponents = impl_Hypre_ParCSRVector_AddtoLocalComponents;
  epv->f_GetRow = impl_Hypre_ParCSRVector_GetRow;
  epv->f_SetValues = impl_Hypre_ParCSRVector_SetValues;
  epv->f_AddToLocalComponentsInBlock = 
    impl_Hypre_ParCSRVector_AddToLocalComponentsInBlock;
  epv->f_Axpy = impl_Hypre_ParCSRVector_Axpy;
}

struct Hypre_ParCSRVector__data*
Hypre_ParCSRVector__get_data(Hypre_ParCSRVector self)
{
  return (struct Hypre_ParCSRVector__data*)(self ? self->d_data : NULL);
}

void Hypre_ParCSRVector__set_data(
  Hypre_ParCSRVector self,
  struct Hypre_ParCSRVector__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
