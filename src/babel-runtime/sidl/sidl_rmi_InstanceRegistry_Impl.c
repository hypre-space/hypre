/*
 * File:          sidl_rmi_InstanceRegistry_Impl.c
 * Symbol:        sidl.rmi.InstanceRegistry-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Release:       $Name: V2-4-0b $
 * Revision:      @(#) $Id: sidl_rmi_InstanceRegistry_Impl.c,v 1.7 2007/09/27 19:35:46 painter Exp $
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
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.rmi.InstanceRegistry" (version 0.9.15)
 * 
 *  
 * This singleton class is implemented by Babel's runtime for RMI
 * libraries to invoke methods on server objects.  It maps
 * objectID strings to sidl_BaseClass objects and vice-versa.
 * 
 * The InstanceRegistry creates and returns a unique string when a
 * new object is added to the registry.  When an object's refcount
 * reaches 0 and it is collected, it is removed from the Instance
 * Registry.
 * 
 * Objects are added to the registry in 3 ways:
 * 1) Added to the server's registry when an object is
 * create[Remote]'d.
 * 2) Implicity added to the local registry when an object is
 * passed as an argument in a remote call.
 * 3) A user may manually add a reference to the local registry
 * for publishing purposes.  The user hsould keep a reference
 * to the object.  Currently, the user cannot provide their own
 * objectID, this capability should probably be added.
 */

#include "sidl_rmi_InstanceRegistry_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._includes) */
#include <stdlib.h>
#include "sidl_String.h"
#include "sidl_Exception.h"

#ifdef HAVE_PTHREAD
#include <pthread.h>
static pthread_mutex_t                s_hash_mutex; /*lock for the hashtables*/
static pthread_mutex_t                s_counter_mutex; /*lock for the hashtables*/
#endif /* HAVE_PTHREAD */


static char* s_counter;

static struct hashtable *s_s2ohshtbl; /* Hash table to map strings to objects*/
static struct hashtable *s_o2shshtbl; /* Hash table to map objects to strings*/

static unsigned int
stringhash(void *ky)
{
  unsigned int hash = 5381;
  unsigned int c;
  char* str = (char*)ky;

  if(ky != NULL){
    while((c = (*str++)))
      hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    
    return hash;
  } else
    return 0U;
}

/* TODO: Find a better hash function for the BaseClass case*/
static unsigned int
objecthash(void *ky)
{
  return (int) ky;
}

/*Object equivilence function*/
static int 
pointer_equals(void* a, void* b) {
  return a == b;
}

/* next_string generates unique alpha-numeric strings to label 
   objects in the instance registry */
char * next_string(void) {
  int i, len;
  char *str;
  char *ret;
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_counter_mutex));
#endif /* HAVE_PTHREAD */

  str = s_counter;
  while(*str != '\0') {
    if(*str < 'z') {
      if(*str == '9') {
	*str = 'A';
      } else if (*str == 'Z') {
	*str = 'a';
      } else { 
	++(*str);
      }
      ret = sidl_String_strdup(s_counter);
#ifdef HAVE_PTHREAD
      (void)pthread_mutex_unlock(&(s_counter_mutex));
#endif /* HAVE_PTHREAD */
      return ret;
    } else {
      *str='0';
      ++str;
    }
  }
  len = sidl_String_strlen(s_counter);
  sidl_String_free(s_counter);
  len <<= 1;
  s_counter=sidl_String_alloc(len);
  for(i = 0; i < len; ++i)
    s_counter[i] = '0';
  s_counter[len] = '\0';
  ret = sidl_String_strdup(s_counter);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_counter_mutex));
#endif /* HAVE_PTHREAD */
  return ret;
}

static void
rmi_InstanceRegistry_cleanup(void)
{
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_lock(&(s_counter_mutex));
#endif /* HAVE_PTHREAD */
  if (s_counter) {
    sidl_String_free(s_counter);
    s_counter = NULL;
  }
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_unlock(&(s_counter_mutex));
#endif /* HAVE_PTHREAD */
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
     if (s_s2ohshtbl) {
       hashtable_destroy(s_s2ohshtbl, 0);
       s_s2ohshtbl = NULL;
     }
     if (s_o2shshtbl) {
       hashtable_destroy(s_o2shshtbl, 0);
       s_o2shshtbl = NULL;
     }
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
}

/* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_InstanceRegistry__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._load) */
  /* Insert the implementation of the static class initializer method here... */
  int i = 0;

#ifdef HAVE_PTHREAD
  (void)pthread_mutex_init(&(s_hash_mutex), NULL);
  (void)pthread_mutex_init(&(s_counter_mutex), NULL);
#endif /* HAVE_PTHREAD */


  /* Since this should really only happen once, before any threads have started do I need mutexs?*/
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_lock(&(s_counter_mutex));
#endif /* HAVE_PTHREAD */
  s_counter = (char*)sidl_String_alloc(4);
  for(i = 0; i<4; ++i) {
    s_counter[i] = '0';
  }
  s_counter[4] = '\0';
#ifdef HAVE_PTHREAD
     (void)pthread_mutex_unlock(&(s_counter_mutex));
#endif /* HAVE_PTHREAD */


#ifdef HAVE_PTHREAD
     (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  s_s2ohshtbl = create_hashtable(16, stringhash, 
			       (int(*)(void*,void*))sidl_String_equals, FALSE);
  s_o2shshtbl = create_hashtable(16, objecthash, 
			       (int(*)(void*,void*))pointer_equals, FALSE);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */

  (void)atexit(rmi_InstanceRegistry_cleanup);
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_InstanceRegistry__ctor(
  /* in */ sidl_rmi_InstanceRegistry self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._ctor) */

  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_InstanceRegistry__ctor2(
  /* in */ sidl_rmi_InstanceRegistry self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._ctor2) */
  /* Insert-Code-Here {sidl.rmi.InstanceRegistry._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_InstanceRegistry__dtor(
  /* in */ sidl_rmi_InstanceRegistry self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._dtor) */

  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._dtor) */
  }
}

/*
 *  
 * Register an instance of a class.
 * 
 * the registry will return an objectID string guaranteed to be
 * unique for the lifetime of the process
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_registerInstance"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_rmi_InstanceRegistry_registerInstance(
  /* in */ sidl_BaseClass instance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.registerInstance) */
  char* key = NULL;
  /* 
   *   DUE TO THE FACT THAT getClassInfo DOES NOT WORK FOR REMOTE OBJECT YET, THIS
   * CODE IS LEFT ON THE SHELF.  WHEN CLASSINFO IS FIXED, TRY THIS CODE:
   * 
   * We create an identifing name for the class from the classname + unique string
   *  sidl_ClassInfo clsinfo = sidl_BaseClass_getClassInfo(instance);
   *   char * clsName = sidl_ClassInfo_getName(clsinfo);
   *  char * instName = sidl_String_concat2(clsName,next_string(s_counter));
   *  sidl_String_free(clsName);
   *  sidl_ClassInfo_deleteRef(clsinfo);
   *  
   *  hashtable_insert(hshtbl, (void*)instName, (void*)instance);
   *  return sidl_String_strdup(instName);
   *  
   *  UNTIL THEN, WE USE THIS CODE:
   */

  
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  key = (char*) hashtable_search(s_o2shshtbl, (void*)instance);
  if(!key) {
    key = next_string();
    hashtable_insert(s_s2ohshtbl, (void*)key, (void*)instance);
    hashtable_insert(s_o2shshtbl, (void*)instance, (void*)key);
  }
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  
  return sidl_String_strdup(key);

  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.registerInstance) */
  }
}

/*
 *  
 * Register an instance of a class with the given instanceID
 * 
 * If a different object already exists in registry under
 * the supplied name, a false is returned, if the object was 
 * successfully registered, true is returned.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_registerInstanceByString"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_rmi_InstanceRegistry_registerInstanceByString(
  /* in */ sidl_BaseClass instance,
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.registerInstanceByString) */
  sidl_BaseClass bc = NULL;
  char * key = NULL;
  char * tmpkey = NULL;

#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  bc = (sidl_BaseClass) hashtable_search(s_s2ohshtbl, (void*)instanceID);
  key = (char*) instanceID;

  if(bc != NULL && instance != bc) {
    do {
      /* Create a new key that's (almost) certainly unique, by combining the
       * the user requested instanceID, which some other object has, and putting
       * a unique string provided by next_string on the end of it.*/
      tmpkey = next_string();
      key = sidl_String_concat2(instanceID, tmpkey);
      sidl_String_free(tmpkey);
      
    } while (hashtable_search(s_s2ohshtbl, (void*)key));    
    hashtable_insert(s_s2ohshtbl, (void*)key, (void*)instance);
    hashtable_insert(s_o2shshtbl, (void*)instance, (void*)key);
  }
  
  /* If there was no object of that name, add this one.*/
  if(bc == NULL) {
    key = sidl_String_strdup(instanceID);
    hashtable_insert(s_s2ohshtbl, (void*)key, (void*)instance);
    hashtable_insert(s_o2shshtbl, (void*)instance, (void*)key);
  }

  /*If the object was already in the registry, we have done nothing*/
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  return sidl_String_strdup(key);
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.registerInstanceByString) */
  }
}

/*
 *  
 * returns a handle to the class based on the unique objectID
 * string, (null if the handle isn't in the table)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_getInstanceByString"

#ifdef __cplusplus
extern "C"
#endif
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_getInstanceByString(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.getInstanceByString) */
  sidl_BaseClass bc = NULL;
  
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  bc = (sidl_BaseClass) hashtable_search(s_s2ohshtbl, (void*)instanceID);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  
  if(bc != NULL) {
    sidl_BaseClass_addRef(bc, _ex);
  }
  return bc;
  
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.getInstanceByString) */
  }
}

/*
 *  
 * takes a class and returns the objectID string associated
 * with it.  (null if the handle isn't in the table)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_getInstanceByClass"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_rmi_InstanceRegistry_getInstanceByClass(
  /* in */ sidl_BaseClass instance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.getInstanceByClass) */
  char* str = NULL;
  
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  str = (char*) hashtable_search(s_o2shshtbl, (void*)instance);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  
  if(str == NULL) {
    return NULL;
  }
  return str;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.getInstanceByClass) */
  }
}

/*
 *  
 * removes an instance from the table based on its objectID
 * string..  returns a pointer to the object, which must be
 * destroyed.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_removeInstanceByString"

#ifdef __cplusplus
extern "C"
#endif
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_removeInstanceByString(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.removeInstanceByString) */
  sidl_BaseClass bc = NULL;
  char* str = NULL;
  
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  if (s_s2ohshtbl) { /* might be NULL during process shutdown */
    bc = (sidl_BaseClass) hashtable_remove(s_s2ohshtbl, (void*)instanceID);
    
    if(bc) {
      if (s_o2shshtbl) { /* might be NULL during process shutdown */
        /* Should be removed from both tables*/
        str = (char*) hashtable_remove(s_o2shshtbl, (void*)bc); 
	sidl_String_free(str);
      }
    }
  }
  
  
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  if(bc != NULL) {
    sidl_BaseClass_addRef(bc, _ex);
  }
  return bc;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.removeInstanceByString) */
  }
}

/*
 *  
 * removes an instance from the table based on its BaseClass
 * pointer.  returns the objectID string, which much be freed.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_removeInstanceByClass"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_rmi_InstanceRegistry_removeInstanceByClass(
  /* in */ sidl_BaseClass instance,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.removeInstanceByClass) */
  char* str = NULL;
  
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */

  /* It's possible to have multiple names to one object, so keep pulling 
   * out names until we get them all */
  if (s_o2shshtbl) { /* might be NULL during process shutdown */
    do {
      sidl_String_free(str); /* free is a no-op if str is NULL*/
      str = (char*) hashtable_remove(s_o2shshtbl, (void*)instance);
      
      if(str) {
	if (s_s2ohshtbl) { /* might be NULL during process shutdown */
	  hashtable_remove(s_s2ohshtbl, (void*)str); /* Should be removed from
                                                      both tables*/
	}
      }
    } while(str);
  }
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_hash_mutex));
#endif /* HAVE_PTHREAD */
  return str;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.removeInstanceByClass) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidl_rmi_InstanceRegistry_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidl_rmi_InstanceRegistry_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidl_rmi_InstanceRegistry_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidl_rmi_InstanceRegistry_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_rmi_InstanceRegistry__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_InstanceRegistry__connectI(url, ar, _ex);
}
struct sidl_rmi_InstanceRegistry__object* 
  impl_sidl_rmi_InstanceRegistry_fcast_sidl_rmi_InstanceRegistry(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_rmi_InstanceRegistry__cast(bi, _ex);
}
