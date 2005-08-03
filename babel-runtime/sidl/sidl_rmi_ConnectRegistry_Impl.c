/*
 * File:          sidl_rmi_ConnectRegistry_Impl.c
 * Symbol:        sidl.rmi.ConnectRegistry-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
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
 * babel-version = 0.10.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.rmi.ConnectRegistry" (version 0.9.3)
 * 
 * This singleton class is implemented by Babel's runtime for to allow RMI 
 * downcasting of objects.  When we downcast an RMI object, we may be required to
 * create a new derived class object with a connect function.  We store all the
 * connect functions in this table for easy access.
 */

#include "sidl_rmi_ConnectRegistry_Impl.h"

#line 54 "../../../babel/runtime/sidl/sidl_rmi_ConnectRegistry_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._includes) */
#include "sidl_String.h"
static struct hashtable *hshtbl;

/*
 *  We are, of course, assuming that the key is actually a string.
 */
static unsigned int
hashfromkey(void *ky)
{
  unsigned long hash = 5381;
  int c;
  char* str = (char*)ky;

  if(ky != 0){
    while(c = (*str++))
      hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    
    return hash;
  } else
    return 0;
}

/* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._includes) */
#line 79 "sidl_rmi_ConnectRegistry_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ConnectRegistry__load(
  void)
{
#line 93 "../../../babel/runtime/sidl/sidl_rmi_ConnectRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._load) */
  hshtbl = create_hashtable(16, hashfromkey, (int(*)(void*,void*))sidl_String_equals);
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._load) */
#line 99 "sidl_rmi_ConnectRegistry_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ConnectRegistry__ctor(
  /* in */ sidl_rmi_ConnectRegistry self)
{
#line 111 "../../../babel/runtime/sidl/sidl_rmi_ConnectRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._ctor) */
  /* Insert-Code-Here {sidl.rmi.ConnectRegistry._ctor} (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._ctor) */
#line 119 "sidl_rmi_ConnectRegistry_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ConnectRegistry__dtor(
  /* in */ sidl_rmi_ConnectRegistry self)
{
#line 130 "../../../babel/runtime/sidl/sidl_rmi_ConnectRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._dtor) */
  /* Insert-Code-Here {sidl.rmi.ConnectRegistry._dtor} (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._dtor) */
#line 140 "sidl_rmi_ConnectRegistry_Impl.c"
}

/*
 * register an instance of a class
 *  the registry will return a string guaranteed to be unique for
 *  the lifetime of the process
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry_registerConnect"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ConnectRegistry_registerConnect(
  /* in */ const char* key,
  /* in */ void* func)
{
#line 152 "../../../babel/runtime/sidl/sidl_rmi_ConnectRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry.registerConnect) */
  hashtable_insert(hshtbl, (void*)key, (void*)func);
  return;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry.registerConnect) */
#line 165 "sidl_rmi_ConnectRegistry_Impl.c"
}

/*
 * returns a handle to the class based on the unique string
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry_getConnect"

#ifdef __cplusplus
extern "C"
#endif
void*
impl_sidl_rmi_ConnectRegistry_getConnect(
  /* in */ const char* key)
{
#line 172 "../../../babel/runtime/sidl/sidl_rmi_ConnectRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry.getConnect) */
  void * func = NULL;
  func = hashtable_search(hshtbl, (void*)key);
  /* If not found, returns NULL*/
  return func;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry.getConnect) */
#line 189 "sidl_rmi_ConnectRegistry_Impl.c"
}

/*
 * returns a handle to the class based on the unique string
 * and removes the instance from the table.  
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry_removeConnect"

#ifdef __cplusplus
extern "C"
#endif
void*
impl_sidl_rmi_ConnectRegistry_removeConnect(
  /* in */ const char* key)
{
#line 195 "../../../babel/runtime/sidl/sidl_rmi_ConnectRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry.removeConnect) */
  void * func = NULL;
  func = (sidl_BaseClass) hashtable_remove(hshtbl, (void*)key);
  /* If not found, returns NULL*/
  return func;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry.removeConnect) */
#line 214 "sidl_rmi_ConnectRegistry_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_ConnectRegistry__connect(url, _ex);
}
char * impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_rmi_ConnectRegistry(struct 
  sidl_rmi_ConnectRegistry__object* obj) {
  return sidl_rmi_ConnectRegistry__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_rmi_ConnectRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
