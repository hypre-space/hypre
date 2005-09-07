/*
 * File:          sidl_ClassInfoI_Impl.c
 * Symbol:        sidl.ClassInfoI-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.ClassInfoI
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
 * babel-version = 0.10.10
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.ClassInfoI" (version 0.9.3)
 * 
 * An implementation of the <code>ClassInfo</code> interface. This provides
 * methods to set all the attributes that are read-only in the
 * <code>ClassInfo</code> interface.
 */

#include "sidl_ClassInfoI_Impl.h"

#line 53 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI._includes) */
#include <stdlib.h>
#include <stdio.h>
#include "sidl_String.h"
/* DO-NOT-DELETE splicer.end(sidl.ClassInfoI._includes) */
#line 59 "sidl_ClassInfoI_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_ClassInfoI__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_ClassInfoI__load(
  void)
{
#line 73 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI._load) */
#line 79 "sidl_ClassInfoI_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_ClassInfoI__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_ClassInfoI__ctor(
  /* in */ sidl_ClassInfoI self)
{
#line 91 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI._ctor) */
  struct sidl_ClassInfoI__data *data = (struct sidl_ClassInfoI__data*)
    malloc(sizeof(struct sidl_ClassInfoI__data));
  if (data) {
    data->d_IOR_major = data->d_IOR_minor = -1;
    data->d_classname = NULL;
  }
  sidl_ClassInfoI__set_data(self, data);
  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI._ctor) */
#line 105 "sidl_ClassInfoI_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_ClassInfoI__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_ClassInfoI__dtor(
  /* in */ sidl_ClassInfoI self)
{
#line 116 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI._dtor) */
  struct sidl_ClassInfoI__data *data = sidl_ClassInfoI__get_data(self);
  if (data) {
    sidl_String_free(data->d_classname);
    free((void *)data);
  }
  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI._dtor) */
#line 130 "sidl_ClassInfoI_Impl.c"
}

/*
 * Set the name of the class.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_ClassInfoI_setName"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_ClassInfoI_setName(
  /* in */ sidl_ClassInfoI self,
  /* in */ const char* name)
{
#line 140 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI.setName) */
  struct sidl_ClassInfoI__data *data = sidl_ClassInfoI__get_data(self);
  if (data) {
    sidl_String_free(data->d_classname);
    data->d_classname = sidl_String_strdup(name);
  }
  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI.setName) */
#line 156 "sidl_ClassInfoI_Impl.c"
}

/*
 * Set the IOR major and minor version numbers.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_ClassInfoI_setIORVersion"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_ClassInfoI_setIORVersion(
  /* in */ sidl_ClassInfoI self,
  /* in */ int32_t major,
  /* in */ int32_t minor)
{
#line 165 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI.setIORVersion) */
  struct sidl_ClassInfoI__data *data = sidl_ClassInfoI__get_data(self);
  if (data) {
    data->d_IOR_major = major;
    data->d_IOR_minor = minor;
  }

  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI.setIORVersion) */
#line 184 "sidl_ClassInfoI_Impl.c"
}

/*
 * Return the name of the class.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_ClassInfoI_getName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_ClassInfoI_getName(
  /* in */ sidl_ClassInfoI self)
{
#line 189 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI.getName) */
  struct sidl_ClassInfoI__data *data = sidl_ClassInfoI__get_data(self);
  return sidl_String_strdup(data ? data->d_classname : NULL);
  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI.getName) */
#line 206 "sidl_ClassInfoI_Impl.c"
}

/*
 * Get the version of the intermediate object representation.
 * This will be in the form of major_version.minor_version.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_ClassInfoI_getIORVersion"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_ClassInfoI_getIORVersion(
  /* in */ sidl_ClassInfoI self)
{
#line 210 "../../../babel/runtime/sidl/sidl_ClassInfoI_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.ClassInfoI.getIORVersion) */
  int32_t major, minor;
  char buf[34];
  struct sidl_ClassInfoI__data *data = sidl_ClassInfoI__get_data(self);
  major = (data ? data->d_IOR_major : -1);
  minor = (data ? data->d_IOR_minor : -1);
  sprintf(buf, "%d.%d", major, minor);
  return sidl_String_strdup(buf);
  /* DO-NOT-DELETE splicer.end(sidl.ClassInfoI.getIORVersion) */
#line 234 "sidl_ClassInfoI_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_ClassInfoI__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfoI(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfoI__connect(url, _ex);
}
char * impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfoI(struct 
  sidl_ClassInfoI__object* obj) {
  return sidl_ClassInfoI__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_ClassInfoI_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_ClassInfoI_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidl_ClassInfoI_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_ClassInfoI_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
