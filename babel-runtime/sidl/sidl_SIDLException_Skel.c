/*
 * File:          sidl_SIDLException_Skel.c
 * Symbol:        sidl.SIDLException-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side glue code for sidl.SIDLException
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
 * babel-version = 0.10.4
 */

#include "sidl_SIDLException_IOR.h"
#include "sidl_SIDLException.h"
#include <stddef.h>

extern
void
impl_sidl_SIDLException__load(
  void);

extern
void
impl_sidl_SIDLException__ctor(
  /* in */ sidl_SIDLException self);

extern
void
impl_sidl_SIDLException__dtor(
  /* in */ sidl_SIDLException self);

extern struct sidl_SIDLException__object* 
  impl_sidl_SIDLException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_SIDLException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
char*
impl_sidl_SIDLException_getNote(
  /* in */ sidl_SIDLException self);

extern
void
impl_sidl_SIDLException_setNote(
  /* in */ sidl_SIDLException self,
  /* in */ const char* message);

extern
char*
impl_sidl_SIDLException_getTrace(
  /* in */ sidl_SIDLException self);

extern
void
impl_sidl_SIDLException_addLine(
  /* in */ sidl_SIDLException self,
  /* in */ const char* traceline);

extern
void
impl_sidl_SIDLException_add(
  /* in */ sidl_SIDLException self,
  /* in */ const char* filename,
  /* in */ int32_t lineno,
  /* in */ const char* methodname);

extern struct sidl_SIDLException__object* 
  impl_sidl_SIDLException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidl_SIDLException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidl_SIDLException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidl_SIDLException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidl_SIDLException__set_epv(struct sidl_SIDLException__epv *epv)
{
  epv->f__ctor = impl_sidl_SIDLException__ctor;
  epv->f__dtor = impl_sidl_SIDLException__dtor;
  epv->f_getNote = impl_sidl_SIDLException_getNote;
  epv->f_setNote = impl_sidl_SIDLException_setNote;
  epv->f_getTrace = impl_sidl_SIDLException_getTrace;
  epv->f_addLine = impl_sidl_SIDLException_addLine;
  epv->f_add = impl_sidl_SIDLException_add;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidl_SIDLException__call_load(void) { 
  impl_sidl_SIDLException__load();
}
struct sidl_SIDLException__object* 
  skel_sidl_SIDLException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_SIDLException_fconnect_sidl_SIDLException(url, _ex);
}

char* skel_sidl_SIDLException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) { 
  return impl_sidl_SIDLException_fgetURL_sidl_SIDLException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidl_SIDLException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_SIDLException_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidl_SIDLException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidl_SIDLException_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidl_SIDLException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_SIDLException_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidl_SIDLException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidl_SIDLException_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseException__object* 
  skel_sidl_SIDLException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_SIDLException_fconnect_sidl_BaseException(url, _ex);
}

char* skel_sidl_SIDLException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) { 
  return impl_sidl_SIDLException_fgetURL_sidl_BaseException(obj);
}

struct sidl_BaseClass__object* 
  skel_sidl_SIDLException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidl_SIDLException_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidl_SIDLException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidl_SIDLException_fgetURL_sidl_BaseClass(obj);
}

struct sidl_SIDLException__data*
sidl_SIDLException__get_data(sidl_SIDLException self)
{
  return (struct sidl_SIDLException__data*)(self ? self->d_data : NULL);
}

void sidl_SIDLException__set_data(
  sidl_SIDLException self,
  struct sidl_SIDLException__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
