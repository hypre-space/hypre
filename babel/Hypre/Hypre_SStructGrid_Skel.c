/*
 * File:          Hypre_SStructGrid_Skel.c
 * Symbol:        Hypre.SStructGrid-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 914
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_SStructGrid_IOR.h"
#include "Hypre_SStructGrid.h"
#include <stddef.h>

extern void
impl_Hypre_SStructGrid__ctor(
  Hypre_SStructGrid);

extern void
impl_Hypre_SStructGrid__dtor(
  Hypre_SStructGrid);

extern int32_t
impl_Hypre_SStructGrid_SetNumDimParts(
  Hypre_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructGrid_SetExtents(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructGrid_SetVariable(
  Hypre_SStructGrid,
  int32_t,
  int32_t,
  enum Hypre_SStructVariable__enum);

extern int32_t
impl_Hypre_SStructGrid_AddVariable(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  enum Hypre_SStructVariable__enum);

extern int32_t
impl_Hypre_SStructGrid_SetNeighborBox(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  int32_t,
  struct SIDL_int__array*,
  struct SIDL_int__array*,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructGrid_AddUnstructuredPart(
  Hypre_SStructGrid,
  int32_t,
  int32_t);

extern int32_t
impl_Hypre_SStructGrid_SetPeriodic(
  Hypre_SStructGrid,
  int32_t,
  struct SIDL_int__array*);

extern int32_t
impl_Hypre_SStructGrid_SetNumGhost(
  Hypre_SStructGrid,
  struct SIDL_int__array*);

void
Hypre_SStructGrid__set_epv(struct Hypre_SStructGrid__epv *epv)
{
  epv->f__ctor = impl_Hypre_SStructGrid__ctor;
  epv->f__dtor = impl_Hypre_SStructGrid__dtor;
  epv->f_SetNumDimParts = impl_Hypre_SStructGrid_SetNumDimParts;
  epv->f_SetExtents = impl_Hypre_SStructGrid_SetExtents;
  epv->f_SetVariable = impl_Hypre_SStructGrid_SetVariable;
  epv->f_AddVariable = impl_Hypre_SStructGrid_AddVariable;
  epv->f_SetNeighborBox = impl_Hypre_SStructGrid_SetNeighborBox;
  epv->f_AddUnstructuredPart = impl_Hypre_SStructGrid_AddUnstructuredPart;
  epv->f_SetPeriodic = impl_Hypre_SStructGrid_SetPeriodic;
  epv->f_SetNumGhost = impl_Hypre_SStructGrid_SetNumGhost;
}

struct Hypre_SStructGrid__data*
Hypre_SStructGrid__get_data(Hypre_SStructGrid self)
{
  return (struct Hypre_SStructGrid__data*)(self ? self->d_data : NULL);
}

void Hypre_SStructGrid__set_data(
  Hypre_SStructGrid self,
  struct Hypre_SStructGrid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
