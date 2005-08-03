/*
 * File:          sidl_InvViolation_Impl.c
 * Symbol:        sidl.InvViolation-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.InvViolation
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
 * babel-version = 0.10.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.InvViolation" (version 0.9.3)
 * 
 * <code>InvViolation</code> provides the basic marker for 
 * a invariant exception.
 */

#include "sidl_InvViolation_Impl.h"

#line 52 "../../../babel/runtime/sidl/sidl_InvViolation_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.InvViolation._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(sidl.InvViolation._includes) */
#line 56 "sidl_InvViolation_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_InvViolation__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_InvViolation__load(
  void)
{
#line 70 "../../../babel/runtime/sidl/sidl_InvViolation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.InvViolation._load) */
  /* Insert implementation here: sidl.InvViolation._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidl.InvViolation._load) */
#line 76 "sidl_InvViolation_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_InvViolation__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_InvViolation__ctor(
  /* in */ sidl_InvViolation self)
{
#line 88 "../../../babel/runtime/sidl/sidl_InvViolation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.InvViolation._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidl.InvViolation._ctor) */
#line 96 "sidl_InvViolation_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_InvViolation__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_InvViolation__dtor(
  /* in */ sidl_InvViolation self)
{
#line 107 "../../../babel/runtime/sidl/sidl_InvViolation_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.InvViolation._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidl.InvViolation._dtor) */
#line 117 "sidl_InvViolation_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_SIDLException__object* 
  impl_sidl_InvViolation_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connect(url, _ex);
}
char * impl_sidl_InvViolation_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) {
  return sidl_SIDLException__getURL(obj);
}
struct sidl_InvViolation__object* 
  impl_sidl_InvViolation_fconnect_sidl_InvViolation(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_InvViolation__connect(url, _ex);
}
char * impl_sidl_InvViolation_fgetURL_sidl_InvViolation(struct 
  sidl_InvViolation__object* obj) {
  return sidl_InvViolation__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidl_InvViolation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_InvViolation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_InvViolation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_InvViolation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseException__object* 
  impl_sidl_InvViolation_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseException__connect(url, _ex);
}
char * impl_sidl_InvViolation_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) {
  return sidl_BaseException__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidl_InvViolation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_InvViolation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
