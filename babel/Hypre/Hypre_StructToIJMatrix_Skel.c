/*
 * File:          Hypre_StructToIJMatrix_Skel.c
 * Symbol:        Hypre.StructToIJMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:45 PDT
 * Description:   Server-side glue code for Hypre.StructToIJMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "Hypre_StructToIJMatrix_IOR.h"
#include "Hypre_StructToIJMatrix.h"
#include <stddef.h>

extern void
impl_Hypre_StructToIJMatrix__ctor(
  Hypre_StructToIJMatrix);

extern void
impl_Hypre_StructToIJMatrix__dtor(
  Hypre_StructToIJMatrix);

extern int32_t
impl_Hypre_StructToIJMatrix_Assemble(
  Hypre_StructToIJMatrix);

extern int32_t
impl_Hypre_StructToIJMatrix_GetObject(
  Hypre_StructToIJMatrix,
  SIDL_BaseInterface*);

extern int32_t
impl_Hypre_StructToIJMatrix_Initialize(
  Hypre_StructToIJMatrix);

extern int32_t
impl_Hypre_StructToIJMatrix_SetBoxValues(
  Hypre_StructToIJMatrix,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

extern int32_t
impl_Hypre_StructToIJMatrix_SetCommunicator(
  Hypre_StructToIJMatrix,
  void*);

extern int32_t
impl_Hypre_StructToIJMatrix_SetGrid(
  Hypre_StructToIJMatrix,
  Hypre_StructGrid);

extern int32_t
impl_Hypre_StructToIJMatrix_SetIJMatrix(
  Hypre_StructToIJMatrix,
  Hypre_IJBuildMatrix);

extern int32_t
impl_Hypre_StructToIJMatrix_SetNumGhost(
  Hypre_StructToIJMatrix,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_StructToIJMatrix_SetStencil(
  Hypre_StructToIJMatrix,
  Hypre_StructStencil);

extern int32_t
impl_Hypre_StructToIJMatrix_SetSymmetric(
  Hypre_StructToIJMatrix,
  int32_t);

extern int32_t
impl_Hypre_StructToIJMatrix_SetValues(
  Hypre_StructToIJMatrix,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_double__array*);

void
Hypre_StructToIJMatrix__set_epv(struct Hypre_StructToIJMatrix__epv *epv)
{
  epv->f__ctor = impl_Hypre_StructToIJMatrix__ctor;
  epv->f__dtor = impl_Hypre_StructToIJMatrix__dtor;
  epv->f_SetBoxValues = impl_Hypre_StructToIJMatrix_SetBoxValues;
  epv->f_SetNumGhost = impl_Hypre_StructToIJMatrix_SetNumGhost;
  epv->f_SetStencil = impl_Hypre_StructToIJMatrix_SetStencil;
  epv->f_SetGrid = impl_Hypre_StructToIJMatrix_SetGrid;
  epv->f_SetCommunicator = impl_Hypre_StructToIJMatrix_SetCommunicator;
  epv->f_GetObject = impl_Hypre_StructToIJMatrix_GetObject;
  epv->f_SetSymmetric = impl_Hypre_StructToIJMatrix_SetSymmetric;
  epv->f_SetValues = impl_Hypre_StructToIJMatrix_SetValues;
  epv->f_SetIJMatrix = impl_Hypre_StructToIJMatrix_SetIJMatrix;
  epv->f_Assemble = impl_Hypre_StructToIJMatrix_Assemble;
  epv->f_Initialize = impl_Hypre_StructToIJMatrix_Initialize;
}

struct Hypre_StructToIJMatrix__data*
Hypre_StructToIJMatrix__get_data(Hypre_StructToIJMatrix self)
{
  return (struct Hypre_StructToIJMatrix__data*)(self ? self->d_data : NULL);
}

void Hypre_StructToIJMatrix__set_data(
  Hypre_StructToIJMatrix self,
  struct Hypre_StructToIJMatrix__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
