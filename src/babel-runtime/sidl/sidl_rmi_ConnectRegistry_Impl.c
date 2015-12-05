/*
 * File:          sidl_rmi_ConnectRegistry_Impl.c
 * Symbol:        sidl.rmi.ConnectRegistry-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_rmi_ConnectRegistry_Impl.c,v 1.6 2006/08/29 22:29:51 painter Exp $
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
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.rmi.ConnectRegistry" (version 0.9.15)
 * 
 *  
 * This singleton class is implemented by Babel's runtime for to
 * allow RMI downcasting of objects.  When we downcast an RMI
 * object, we may be required to create a new derived class object
 * with a connect function.  We store all the connect functions in
 * this table for easy access.
 * 
 * This Class is for Babel internal use only.
 */

#include "sidl_rmi_ConnectRegistry_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._includes) */
#include "sidl_String.h"
#include <stdlib.h>
#include "sidlOps.h"

#ifdef HAVE_PTHREAD
#include <pthread.h>
static pthread_mutex_t                s_hash_mutex; /*lock for the hashtables*/
#endif /* HAVE_PTHREAD */

static struct hashtable *s_hshtbl = NULL;


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
    while((c = (*str++)))
      hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    
    return hash;
  } else
    return 0;
}

static void
cleanupRegistry(void *ignored)
{
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  if (s_hshtbl) {
    hashtable_destroy(s_hshtbl, 0);
    s_hshtbl = NULL;
  }
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
}

/* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._load) */
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_init(&(s_hash_mutex), NULL);
#endif /* HAVE_PTHREAD */

#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  s_hshtbl = create_hashtable(16, hashfromkey, (int(*)(void*,void*))sidl_String_equals, TRUE);
  sidl_atexit(cleanupRegistry, NULL);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */


  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._load) */
  }
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
  /* in */ sidl_rmi_ConnectRegistry self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._ctor) */
  /* Insert-Code-Here {sidl.rmi.ConnectRegistry._ctor} (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ConnectRegistry__ctor2(
  /* in */ sidl_rmi_ConnectRegistry self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._ctor2) */
  /* Insert-Code-Here {sidl.rmi.ConnectRegistry._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._ctor2) */
  }
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
  /* in */ sidl_rmi_ConnectRegistry self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry._dtor) */
  /* Insert-Code-Here {sidl.rmi.ConnectRegistry._dtor} (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry._dtor) */
  }
}

/*
 *  
 * The key is the SIDL classname the registered connect belongs
 * to.  Multiple registrations under the same key are possible,
 * this must be protected against in the user code.  Babel does
 * this internally with a static boolean.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry_registerConnect"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ConnectRegistry_registerConnect(
  /* in */ const char* key,
  /* in */ void* func,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry.registerConnect) */
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  if (s_hshtbl) {
    char *key_copy = sidl_String_strdup(key);
    hashtable_insert(s_hshtbl, (void*)key_copy, (void*)func);
  }
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  return;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry.registerConnect) */
  }
}

/*
 *  
 * Returns the connect method for the class named in the key
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry_getConnect"

#ifdef __cplusplus
extern "C"
#endif
void*
impl_sidl_rmi_ConnectRegistry_getConnect(
  /* in */ const char* key,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry.getConnect) */
  void * func = NULL;

#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  if (s_hshtbl) func = hashtable_search(s_hshtbl, (void*)key);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  /* If not found, returns NULL*/
  return func;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry.getConnect) */
  }
}

/*
 *  
 * Returns the connect method for the class named in the key,
 * and removes it from the table.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ConnectRegistry_removeConnect"

#ifdef __cplusplus
extern "C"
#endif
void*
impl_sidl_rmi_ConnectRegistry_removeConnect(
  /* in */ const char* key,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ConnectRegistry.removeConnect) */
  void * func = NULL;
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  if (s_hshtbl) 
    func = (sidl_BaseClass) hashtable_remove(s_hshtbl, (void*)key);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */

  /* If not found, returns NULL*/
  return func;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ConnectRegistry.removeConnect) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fconnect_sidl_rmi_ConnectRegistry(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_ConnectRegistry__connectI(url, ar, _ex);
}
struct sidl_rmi_ConnectRegistry__object* 
  impl_sidl_rmi_ConnectRegistry_fcast_sidl_rmi_ConnectRegistry(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_rmi_ConnectRegistry__cast(bi, _ex);
}
