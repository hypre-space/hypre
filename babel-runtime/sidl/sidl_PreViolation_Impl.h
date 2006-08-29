/*
 * File:          sidl_PreViolation_Impl.h
 * Symbol:        sidl.PreViolation-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.PreViolation
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

#ifndef included_sidl_PreViolation_Impl_h
#define included_sidl_PreViolation_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
#ifndef included_sidl_PreViolation_h
#include "sidl_PreViolation.h"
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
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidl.PreViolation._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidl.PreViolation._includes) */

/*
 * Private data for class sidl.PreViolation
 */

struct sidl_PreViolation__data {
  /* DO-NOT-DELETE splicer.begin(sidl.PreViolation._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidl.PreViolation._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidl_PreViolation__data*
sidl_PreViolation__get_data(
  sidl_PreViolation);

extern void
sidl_PreViolation__set_data(
  sidl_PreViolation,
  struct sidl_PreViolation__data*);

extern
void
impl_sidl_PreViolation__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_PreViolation__ctor(
  /* in */ sidl_PreViolation self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_PreViolation__ctor2(
  /* in */ sidl_PreViolation self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidl_PreViolation__dtor(
  /* in */ sidl_PreViolation self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_PreViolation_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidl_PreViolation_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_PreViolation_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_PreViolation_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_PreViolation_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_PreViolation__object* 
  impl_sidl_PreViolation_fconnect_sidl_PreViolation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_PreViolation__object* 
  impl_sidl_PreViolation_fcast_sidl_PreViolation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_PreViolation_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_PreViolation_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_PreViolation_fconnect_sidl_SIDLException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_PreViolation_fcast_sidl_SIDLException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_PreViolation_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_PreViolation_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_PreViolation_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_PreViolation_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_PreViolation_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_PreViolation_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidl_PreViolation_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidl_PreViolation_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_PreViolation_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidl_PreViolation_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_PreViolation_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidl_PreViolation_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_PreViolation__object* 
  impl_sidl_PreViolation_fconnect_sidl_PreViolation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_PreViolation__object* 
  impl_sidl_PreViolation_fcast_sidl_PreViolation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_PreViolation_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidl_PreViolation_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_PreViolation_fconnect_sidl_SIDLException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidl_PreViolation_fcast_sidl_SIDLException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_PreViolation_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidl_PreViolation_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_PreViolation_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidl_PreViolation_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_PreViolation_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidl_PreViolation_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
