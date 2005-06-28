/*
 * File:          sidl_rmi_ProtocolFactory_Impl.h
 * Symbol:        sidl.rmi.ProtocolFactory-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.rmi.ProtocolFactory
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

#ifndef included_sidl_rmi_ProtocolFactory_Impl_h
#define included_sidl_rmi_ProtocolFactory_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
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
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory._includes) */

/*
 * Private data for class sidl.rmi.ProtocolFactory
 */

struct sidl_rmi_ProtocolFactory__data {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory._data) */
  char ** reg; /* reg[2*i] is associated w/ reg[2*i+1] */
  int len; /* len/2 entries, len always even */
  int maxlen; /* size of buffer */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_rmi_ProtocolFactory__data*
sidl_rmi_ProtocolFactory__get_data(
  sidl_rmi_ProtocolFactory);

extern void
sidl_rmi_ProtocolFactory__set_data(
  sidl_rmi_ProtocolFactory,
  struct sidl_rmi_ProtocolFactory__data*);

extern
void
impl_sidl_rmi_ProtocolFactory__load(
  void);

extern
void
impl_sidl_rmi_ProtocolFactory__ctor(
  /* in */ sidl_rmi_ProtocolFactory self);

extern
void
impl_sidl_rmi_ProtocolFactory__dtor(
  /* in */ sidl_rmi_ProtocolFactory self);

/*
 * User-defined object methods
 */

extern
sidl_bool
impl_sidl_rmi_ProtocolFactory_addProtocol(
  /* in */ const char* prefix,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidl_rmi_ProtocolFactory_getProtocol(
  /* in */ const char* prefix,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidl_rmi_ProtocolFactory_deleteProtocol(
  /* in */ const char* prefix,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_InstanceHandle
impl_sidl_rmi_ProtocolFactory_createInstance(
  /* in */ const char* url,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_InstanceHandle
impl_sidl_rmi_ProtocolFactory_connectInstance(
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(struct 
  sidl_rmi_ProtocolFactory__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(struct 
  sidl_rmi_ProtocolFactory__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
