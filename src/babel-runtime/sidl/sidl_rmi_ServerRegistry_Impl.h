/*
 * File:          sidl_rmi_ServerRegistry_Impl.h
 * Symbol:        sidl.rmi.ServerRegistry-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_rmi_ServerRegistry_Impl.h,v 1.1 2006/08/29 23:26:42 painter Exp $
 * Description:   Server-side implementation for sidl.rmi.ServerRegistry
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

#ifndef included_sidl_rmi_ServerRegistry_Impl_h
#define included_sidl_rmi_ServerRegistry_Impl_h

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
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_rmi_ServerInfo_h
#include "sidl_rmi_ServerInfo.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.rmi.ServerRegistry._includes) */
/* Insert-Code-Here {sidl.rmi.ServerRegistry._includes} (include files) */
/* DO-NOT-DELETE splicer.end(sidl.rmi.ServerRegistry._includes) */

/*
 * Private data for class sidl.rmi.ServerRegistry
 */

struct sidl_rmi_ServerRegistry__data {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ServerRegistry._data) */
  /* Insert-Code-Here {sidl.rmi.ServerRegistry._data} (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ServerRegistry._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_rmi_ServerRegistry__data*
sidl_rmi_ServerRegistry__get_data(
  sidl_rmi_ServerRegistry);

extern void
sidl_rmi_ServerRegistry__set_data(
  sidl_rmi_ServerRegistry,
  struct sidl_rmi_ServerRegistry__data*);

extern
void
impl_sidl_rmi_ServerRegistry__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ServerRegistry__ctor(
  /* in */ sidl_rmi_ServerRegistry self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ServerRegistry__ctor2(
  /* in */ sidl_rmi_ServerRegistry self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ServerRegistry__dtor(
  /* in */ sidl_rmi_ServerRegistry self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
void
impl_sidl_rmi_ServerRegistry_registerServer(
  /* in */ sidl_rmi_ServerInfo si,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_rmi_ServerRegistry_getServerURL(
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_rmi_ServerRegistry_isLocalObject(
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern
struct sidl_io_Serializable__array*
impl_sidl_rmi_ServerRegistry_getExceptions(
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerRegistry__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_rmi_ServerRegistry(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerRegistry__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_rmi_ServerRegistry(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerRegistry__object* 
  impl_sidl_rmi_ServerRegistry_fconnect_sidl_rmi_ServerRegistry(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerRegistry__object* 
  impl_sidl_rmi_ServerRegistry_fcast_sidl_rmi_ServerRegistry(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
