/*
 * File:          sidl_rmi_ConnectRegistry_Impl.h
 * Symbol:        sidl.rmi.ConnectRegistry-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V1-14-0b $
 * Revision:      @(#) $Id: sidl_rmi_ConnectRegistry_Impl.h,v 1.6 2006/08/29 22:29:51 painter Exp $
 * Description:   Server-side implementation for sidl.rmi.ConnectRegistry
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_sidl_rmi_ConnectRegistry_Impl_h
#define included_sidl_rmi_ConnectRegistry_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._includes) */
#include "sidl_hashtable.h"
/* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._includes) */

/*
 * Private data for class sidl.rmi.ConnectRegistry
 */

struct sidl_rmi_ConnectRegistry__data {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._data) */
  /* Insert-Code-Here {sidl.rmi.ConnectRegistry._data} (private data members) */
  struct hashtable *hshtbl;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_rmi_ConnectRegistry__data*
sidl_rmi_ConnectRegistry__get_data(
  sidl_rmi_ConnectRegistry);

extern void
sidl_rmi_ConnectRegistry__set_data(
  sidl_rmi_ConnectRegistry,
  struct sidl_rmi_ConnectRegistry__data*);

extern
void
impl_sidl_rmi_ConnectRegistry__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ConnectRegistry__ctor(
  /* in */ sidl_rmi_ConnectRegistry self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ConnectRegistry__ctor2(
  /* in */ sidl_rmi_ConnectRegistry self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ConnectRegistry__dtor(
  /* in */ sidl_rmi_ConnectRegistry self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
void
impl_sidl_rmi_ConnectRegistry_registerConnect(
  /* in */ const char* key,
  /* in */ void* func,
  /* out */ sidl_BaseInterface *_ex);

extern
void*
impl_sidl_rmi_ConnectRegistry_getConnect(
  /* in */ const char* key,
  /* out */ sidl_BaseInterface *_ex);

extern
void*
impl_sidl_rmi_ConnectRegistry_removeConnect(
  /* in */ const char* key,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_rmi_ConnectRegistry(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_rmi_ConnectRegistry(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
