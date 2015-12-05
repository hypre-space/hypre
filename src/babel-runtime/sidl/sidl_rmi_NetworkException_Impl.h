/*
 * File:          sidl_rmi_NetworkException_Impl.h
 * Symbol:        sidl.rmi.NetworkException-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V1-14-0b $
 * Revision:      @(#) $Id: sidl_rmi_NetworkException_Impl.h,v 1.6 2006/08/29 22:29:51 painter Exp $
 * Description:   Server-side implementation for sidl.rmi.NetworkException
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

#ifndef included_sidl_rmi_NetworkException_Impl_h
#define included_sidl_rmi_NetworkException_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_rmi_NetworkException_IOR_h
#include "sidl_rmi_NetworkException_IOR.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
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
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifndef included_sidl_io_IOException_h
#include "sidl_io_IOException.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.rmi.NetworkException._includes) */
/* insert implementation here: sidl.rmi.NetworkException._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidl.rmi.NetworkException._includes) */

/*
 * Private data for class sidl.rmi.NetworkException
 */

struct sidl_rmi_NetworkException__data {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.NetworkException._data) */
  /* insert implementation here: sidl.rmi.NetworkException._data (private data members) */
  int hop_count; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.NetworkException._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_rmi_NetworkException__data*
sidl_rmi_NetworkException__get_data(
  sidl_rmi_NetworkException);

extern void
sidl_rmi_NetworkException__set_data(
  sidl_rmi_NetworkException,
  struct sidl_rmi_NetworkException__data*);

extern void sidl_rmi_NetworkException__superEPV(
struct sidl_io_IOException__epv*);

extern
void
impl_sidl_rmi_NetworkException__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_NetworkException__ctor(
  /* in */ sidl_rmi_NetworkException self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_NetworkException__ctor2(
  /* in */ sidl_rmi_NetworkException self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_NetworkException__dtor(
  /* in */ sidl_rmi_NetworkException self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_SIDLException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_SIDLException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_IOException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_IOException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_IOException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_IOException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_rmi_NetworkException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_rmi_NetworkException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_sidl_rmi_NetworkException_getHopCount(
  /* in */ sidl_rmi_NetworkException self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_NetworkException_packObj(
  /* in */ sidl_rmi_NetworkException self,
  /* in */ sidl_io_Serializer ser,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_rmi_NetworkException_unpackObj(
  /* in */ sidl_rmi_NetworkException self,
  /* in */ sidl_io_Deserializer des,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_SIDLException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_SIDLException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_IOException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_IOException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_IOException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_IOException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_NetworkException_fconnect_sidl_rmi_NetworkException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_NetworkException_fcast_sidl_rmi_NetworkException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
