/*
 * File:          bHYPRE_SStructStencil_Impl.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 */

#include "bHYPRE_SStructStencil_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional includes or other arbitrary code here... */

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

#include <assert.h>
#include "hypre_babel_exception_handler.h"
/*#include "mpi.h"*/
#include "sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructStencil__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._load) */
  /* Insert-Code-Here {bHYPRE.SStructStencil._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructStencil__ctor(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_SStructStencil__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructStencil__data, 1 );
   data -> stencil = NULL;
   bHYPRE_SStructStencil__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructStencil__ctor2(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._ctor2) */
    /* Insert-Code-Here {bHYPRE.SStructStencil._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructStencil__dtor(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil stencil;
   data = bHYPRE_SStructStencil__get_data( self );
   stencil = data -> stencil;
   ierr += HYPRE_SStructStencilDestroy( stencil );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructStencil
impl_bHYPRE_SStructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.Create) */
  /* Insert-Code-Here {bHYPRE.SStructStencil.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructStencil stencil;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil Hstencil;

   stencil = bHYPRE_SStructStencil__create(_ex); SIDL_CHECK(*_ex);
   data = bHYPRE_SStructStencil__get_data( stencil );

   ierr += HYPRE_SStructStencilCreate( ndim, size, &Hstencil );
   data->stencil = Hstencil;

   return stencil;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.Create) */
  }
}

/*
 * Set the number of spatial dimensions and stencil entries.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_SetNumDimSize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetNumDimSize) */
  /* Insert the implementation of the SetNumDimSize method here... */
   /* note: StructStencil does this with two functions, SStruct with one.
      But StructStencil_SetElement and SStructStencil_SetEntry are inherently
      different, and no other stencil classes are expected to exist, so there
      is little point in reconciling this */
 
   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil stencil;

   data = bHYPRE_SStructStencil__get_data( self );
   stencil = data -> stencil;

   ierr += HYPRE_SStructStencilCreate( ndim, size, &stencil );
   data -> stencil = stencil;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetNumDimSize) */
  }
}

/*
 * Set a stencil entry.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_SetEntry"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetEntry) */
  /* Insert the implementation of the SetEntry method here... */
 
   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil Hstencil;

   data = bHYPRE_SStructStencil__get_data( self );
   Hstencil = data -> stencil;

   ierr += HYPRE_SStructStencilSetEntry( Hstencil, entry,
                                         offset, var );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetEntry) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructStencil__connectI(url, ar, _ex);
}
struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fcast_bHYPRE_SStructStencil(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructStencil__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
