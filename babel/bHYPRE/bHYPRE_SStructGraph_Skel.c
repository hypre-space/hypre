/*
 * File:          bHYPRE_SStructGraph_Skel.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030314 14:22:36 PST
 * Generated:     20030314 14:22:39 PST
 * Description:   Server-side glue code for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1010
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#include "bHYPRE_SStructGraph_IOR.h"
#include "bHYPRE_SStructGraph.h"
#include <stddef.h>

extern void
impl_bHYPRE_SStructGraph__ctor(
  bHYPRE_SStructGraph);

extern void
impl_bHYPRE_SStructGraph__dtor(
  bHYPRE_SStructGraph);

extern int32_t
impl_bHYPRE_SStructGraph_SetGrid(
  bHYPRE_SStructGraph,
  bHYPRE_SStructGrid);

extern int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  bHYPRE_SStructGraph,
  int32_t,
  int32_t,
  bHYPRE_SStructStencil);

extern int32_t
impl_bHYPRE_SStructGraph_AddEntries(
  bHYPRE_SStructGraph,
  int32_t,
  struct SIDL_int__array*,
  int32_t,
  int32_t,
  struct SIDL_int__array*,
  int32_t);

void
bHYPRE_SStructGraph__set_epv(struct bHYPRE_SStructGraph__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructGraph__ctor;
  epv->f__dtor = impl_bHYPRE_SStructGraph__dtor;
  epv->f_SetGrid = impl_bHYPRE_SStructGraph_SetGrid;
  epv->f_SetStencil = impl_bHYPRE_SStructGraph_SetStencil;
  epv->f_AddEntries = impl_bHYPRE_SStructGraph_AddEntries;
}

struct bHYPRE_SStructGraph__data*
bHYPRE_SStructGraph__get_data(bHYPRE_SStructGraph self)
{
  return (struct bHYPRE_SStructGraph__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructGraph__set_data(
  bHYPRE_SStructGraph self,
  struct bHYPRE_SStructGraph__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
