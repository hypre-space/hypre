/*
 * File:          Hypre_SStructGraph_Skel.c
 * Symbol:        Hypre.SStructGraph-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side glue code for Hypre.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1032
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "Hypre_SStructGraph_IOR.h"
#include "Hypre_SStructGraph.h"
#include <stddef.h>

extern void
impl_Hypre_SStructGraph__ctor(
  Hypre_SStructGraph);

extern void
impl_Hypre_SStructGraph__dtor(
  Hypre_SStructGraph);

extern int32_t
impl_Hypre_SStructGraph_SetGrid(
  Hypre_SStructGraph,
  Hypre_SStructGrid);

extern int32_t
impl_Hypre_SStructGraph_SetStencil(
  Hypre_SStructGraph,
  int32_t,
  int32_t,
  Hypre_SStructStencil);

extern int32_t
impl_Hypre_SStructGraph_AddEntries(
  Hypre_SStructGraph,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  int32_t);

void
Hypre_SStructGraph__set_epv(struct Hypre_SStructGraph__epv *epv)
{
  epv->f__ctor = impl_Hypre_SStructGraph__ctor;
  epv->f__dtor = impl_Hypre_SStructGraph__dtor;
  epv->f_SetGrid = impl_Hypre_SStructGraph_SetGrid;
  epv->f_SetStencil = impl_Hypre_SStructGraph_SetStencil;
  epv->f_AddEntries = impl_Hypre_SStructGraph_AddEntries;
}

struct Hypre_SStructGraph__data*
Hypre_SStructGraph__get_data(Hypre_SStructGraph self)
{
  return (struct Hypre_SStructGraph__data*)(self ? self->d_data : NULL);
}

void Hypre_SStructGraph__set_data(
  Hypre_SStructGraph self,
  struct Hypre_SStructGraph__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
