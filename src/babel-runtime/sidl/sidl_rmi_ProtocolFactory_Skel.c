/*
 * File:          sidl_rmi_ProtocolFactory_Skel.c
 * Symbol:        sidl.rmi.ProtocolFactory-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V1-14-0b $
 * Revision:      @(#) $Id: sidl_rmi_ProtocolFactory_Skel.c,v 1.5 2006/08/29 22:29:52 painter Exp $
 * Description:   Server-side glue code for sidl.rmi.ProtocolFactory
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
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidl_rmi_ProtocolFactory_IOR.h"
#include "sidl_rmi_ProtocolFactory.h"
#include <stddef.h>

extern
void
impl_sidl_rmi_ProtocolFactory__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ProtocolFactory__ctor(
  /* in */ sidl_rmi_ProtocolFactory self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ProtocolFactory__ctor2(
  /* in */ sidl_rmi_ProtocolFactory self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_ProtocolFactory__dtor(
  /* in */ sidl_rmi_ProtocolFactory self,
  /* out */ sidl_BaseInterface *_ex);

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
  /* in */ sidl_bool ar,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_io_Serializable
impl_sidl_rmi_ProtocolFactory_unserializeInstance(
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_InstanceHandle(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_ProtocolFactory(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_InstanceHandle(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_ProtocolFactory(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_ProtocolFactory__set_epv(struct sidl_rmi_ProtocolFactory__epv *epv)
{
  epv->f__ctor = impl_sidl_rmi_ProtocolFactory__ctor;
  epv->f__ctor2 = impl_sidl_rmi_ProtocolFactory__ctor2;
  epv->f__dtor = impl_sidl_rmi_ProtocolFactory__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
sidl_rmi_ProtocolFactory__set_sepv(struct sidl_rmi_ProtocolFactory__sepv *sepv)
{
  sepv->f_addProtocol = impl_sidl_rmi_ProtocolFactory_addProtocol;
  sepv->f_getProtocol = impl_sidl_rmi_ProtocolFactory_getProtocol;
  sepv->f_deleteProtocol = impl_sidl_rmi_ProtocolFactory_deleteProtocol;
  sepv->f_createInstance = impl_sidl_rmi_ProtocolFactory_createInstance;
  sepv->f_connectInstance = impl_sidl_rmi_ProtocolFactory_connectInstance;
  sepv->f_unserializeInstance = 
    impl_sidl_rmi_ProtocolFactory_unserializeInstance;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_rmi_ProtocolFactory__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidl_rmi_ProtocolFactory__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidl_rmi_ProtocolFactory_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(url, ar,
    _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidl_rmi_ProtocolFactory_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidl_rmi_ProtocolFactory_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_RuntimeException(url, ar,
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidl_rmi_ProtocolFactory_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_io_Serializable(url, ar,
    _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidl_rmi_ProtocolFactory_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_rmi_InstanceHandle__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(url, ar,
    _ex);
}

struct sidl_rmi_InstanceHandle__object* 
  skel_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_InstanceHandle(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_InstanceHandle(bi, _ex);
}

struct sidl_rmi_ProtocolFactory__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(url,
    ar, _ex);
}

struct sidl_rmi_ProtocolFactory__object* 
  skel_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_ProtocolFactory(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_ProtocolFactory(bi, _ex);
}

struct sidl_rmi_ProtocolFactory__data*
sidl_rmi_ProtocolFactory__get_data(sidl_rmi_ProtocolFactory self)
{
  return (struct sidl_rmi_ProtocolFactory__data*)(self ? self->d_data : NULL);
}

void sidl_rmi_ProtocolFactory__set_data(
  sidl_rmi_ProtocolFactory self,
  struct sidl_rmi_ProtocolFactory__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
