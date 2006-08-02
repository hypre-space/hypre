/*
 * File:          bHYPRE_SStructGraph_Impl.h
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_SStructGraph_Impl_h
#define included_bHYPRE_SStructGraph_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
#endif
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_h
#include "bHYPRE_ProblemDefinition.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_bHYPRE_SStructGraph_h
#include "bHYPRE_SStructGraph.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._includes) */
/* Put additional include files here... */

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "HYPRE_sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._includes) */

/*
 * Private data for class bHYPRE.SStructGraph
 */

struct bHYPRE_SStructGraph__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._data) */
  /* Put private data members here... */
   HYPRE_SStructGraph graph;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructGraph__data*
bHYPRE_SStructGraph__get_data(
  bHYPRE_SStructGraph);

extern void
bHYPRE_SStructGraph__set_data(
  bHYPRE_SStructGraph,
  struct bHYPRE_SStructGraph__data*);

extern
void
impl_bHYPRE_SStructGraph__load(
  void);

extern
void
impl_bHYPRE_SStructGraph__ctor(
  /* in */ bHYPRE_SStructGraph self);

extern
void
impl_bHYPRE_SStructGraph__dtor(
  /* in */ bHYPRE_SStructGraph self);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructGraph
impl_bHYPRE_SStructGraph_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructGraph_SetCommGrid(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid);

extern
int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ bHYPRE_SStructStencil stencil);

extern
int32_t
impl_bHYPRE_SStructGraph_AddEntries(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t to_part,
  /* in rarray[dim] */ int32_t* to_index,
  /* in */ int32_t to_var);

extern
int32_t
impl_bHYPRE_SStructGraph_SetObjectType(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t type);

extern
int32_t
impl_bHYPRE_SStructGraph_SetCommunicator(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

extern
int32_t
impl_bHYPRE_SStructGraph_Initialize(
  /* in */ bHYPRE_SStructGraph self);

extern
int32_t
impl_bHYPRE_SStructGraph_Assemble(
  /* in */ bHYPRE_SStructGraph self);

extern struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
