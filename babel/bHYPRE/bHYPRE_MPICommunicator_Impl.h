/*
 * File:          bHYPRE_MPICommunicator_Impl.h
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_bHYPRE_MPICommunicator_Impl_h
#define included_bHYPRE_MPICommunicator_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._includes) */
/* Insert-Code-Here {bHYPRE.MPICommunicator._includes} (include files) */

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

#include "utilities.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._includes) */

/*
 * Private data for class bHYPRE.MPICommunicator
 */

struct bHYPRE_MPICommunicator__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._data) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._data} (private data members) */

   MPI_Comm mpi_comm;

  /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_MPICommunicator__data*
bHYPRE_MPICommunicator__get_data(
  bHYPRE_MPICommunicator);

extern void
bHYPRE_MPICommunicator__set_data(
  bHYPRE_MPICommunicator,
  struct bHYPRE_MPICommunicator__data*);

extern
void
impl_bHYPRE_MPICommunicator__load(
  void);

extern
void
impl_bHYPRE_MPICommunicator__ctor(
  /* in */ bHYPRE_MPICommunicator self);

extern
void
impl_bHYPRE_MPICommunicator__dtor(
  /* in */ bHYPRE_MPICommunicator self);

/*
 * User-defined object methods
 */

extern
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateC(
  /* in */ void* mpi_comm);

extern
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateF(
  /* in */ void* mpi_comm);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
