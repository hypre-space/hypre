/*
 * File:          sidl_rmi_InstanceRegistry_Impl.h
 * Symbol:        sidl.rmi.InstanceRegistry-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.rmi.InstanceRegistry
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
 * babel-version = 0.10.4
 */

#ifndef included_sidl_rmi_InstanceRegistry_Impl_h
#define included_sidl_rmi_InstanceRegistry_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._includes) */
#include "sidl_hashtable.h"
/* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._includes) */

/*
 * Private data for class sidl.rmi.InstanceRegistry
 */

struct sidl_rmi_InstanceRegistry__data {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._data) */
  /* Put private data members here... */
  char* counter;
  struct hashtable *hshtbl;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_rmi_InstanceRegistry__data*
sidl_rmi_InstanceRegistry__get_data(
  sidl_rmi_InstanceRegistry);

extern void
sidl_rmi_InstanceRegistry__set_data(
  sidl_rmi_InstanceRegistry,
  struct sidl_rmi_InstanceRegistry__data*);

extern
void
impl_sidl_rmi_InstanceRegistry__load(
  void);

extern
void
impl_sidl_rmi_InstanceRegistry__ctor(
  /* in */ sidl_rmi_InstanceRegistry self);

extern
void
impl_sidl_rmi_InstanceRegistry__dtor(
  /* in */ sidl_rmi_InstanceRegistry self);

/*
 * User-defined object methods
 */

extern
char*
impl_sidl_rmi_InstanceRegistry_registerInstance(
  /* in */ sidl_BaseClass instance,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_getInstance(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_removeInstance(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_InstanceRegistry__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_InstanceRegistry(struct 
  sidl_rmi_InstanceRegistry__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_rmi_InstanceRegistry__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_InstanceRegistry(struct 
  sidl_rmi_InstanceRegistry__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
