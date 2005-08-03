/*
 * File:          sidl_rmi_InstanceRegistry_Impl.c
 * Symbol:        sidl.rmi.InstanceRegistry-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Release:       $Name$
 * Revision:      @(#) $Id$
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
 * babel-version = 0.10.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.rmi.InstanceRegistry" (version 0.9.3)
 * 
 * This singleton class is implemented by Babel's runtime for RMI libraries to 
 * invoke methods on server objects.  It is assumed that the RMI library
 * has a self-describing stream of data, but the data may be reordered
 * from the natural argument list.
 * 
 * 
 * In the case of the RMI library receiving a self-describing stream
 * and wishing to invoke a method on a class... the RMI library would 
 * make a sequence of calls like:
 * 
 *       sidl_BaseClass bc = sidl_rmi_InstanceRegistry_getInstance( "instanceID" );
 *       sidl_rmi_TypeMap inArgs = sidl_rmi_TypeMap__create();
 *       
 *       sidl_rmi_TypeMap_putDouble( inArgs, "input_val" , 2.0 );
 *       sidl_rmi_TypeMap_putString( inArgs, "input_str", "Hello" );
 *       ...
 *       sidl_rmi_TypeMap ourArgs = sidl_BaseClass_execMethod( bc, "methodName" , t );
 * 
 *       sidl_rmi_Response_unpackBool( i, "_retval", &succeeded );
 *       sidl_rmi_Response_unpackFloat( i, "output_val", &f );
 */

#include "sidl_rmi_InstanceRegistry_Impl.h"

#line 70 "../../../babel/runtime/sidl/sidl_rmi_InstanceRegistry_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._includes) */
#include "sidl_String.h"

static char* counter;
static struct hashtable *hshtbl;

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


/* next_string generates unique alpha-numeric strings to label 
   objects in the instance registry */
char * next_string(char * buf) {
  int i, len;
  char *str = buf;
  while(*str != '\0') {
    if(*str < 'z') {
      if(*str == '9') {
	*str = 'A';
      } else if (*str == 'Z') {
	*str = 'a';
      } else { 
	++(*str);
      }
      return buf;
    } else {
      *str='0';
      ++str;
    }
  }
  len = sidl_String_strlen(buf);
  sidl_String_free(buf);
  buf=sidl_String_alloc(len*2);
  for(i = 0; i <= len*2; ++i)
    buf[i] = '!';
  buf[(len*2)+1] = '\0';
  return buf;
}

/* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._includes) */
#line 124 "sidl_rmi_InstanceRegistry_Impl.c"

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
  void)
{
#line 138 "../../../babel/runtime/sidl/sidl_rmi_InstanceRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._load) */
  /* Insert the implementation of the static class initializer method here... */
  int i = 0;
  counter = (char*)sidl_String_alloc(10);
  for(i = 0; i<10; ++i) {
    counter[i] = '0';
  }
  counter[10] = '\0';
  hshtbl = create_hashtable(16, hashfromkey, (int(*)(void*,void*))sidl_String_equals);
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._load) */
#line 151 "sidl_rmi_InstanceRegistry_Impl.c"
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
  /* in */ sidl_rmi_InstanceRegistry self)
{
#line 163 "../../../babel/runtime/sidl/sidl_rmi_InstanceRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._ctor) */

  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._ctor) */
#line 171 "sidl_rmi_InstanceRegistry_Impl.c"
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
  /* in */ sidl_rmi_InstanceRegistry self)
{
#line 182 "../../../babel/runtime/sidl/sidl_rmi_InstanceRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry._dtor) */
  /*struct sidl_rmi_InstanceRegistry__data *dptr =
     sidl_rmi_InstanceRegistry__get_data(self);
   if(dptr) {
     hashtable_destroy(dptr->hshtbl,0);
     sidl_String_free(dptr->counter);
     free(dptr);
     sidl_rmi_InstanceRegistry__set_data(self, 0);
   }
  */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry._dtor) */
#line 200 "sidl_rmi_InstanceRegistry_Impl.c"
}

/*
 * register an instance of a class
 *  the registry will return a string guaranteed to be unique for
 *  the lifetime of the process
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
#line 212 "../../../babel/runtime/sidl/sidl_rmi_InstanceRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.registerInstance) */

  /* 
    DUE TO THE FACT THAT getClassInfo DOES NOT WORK FOR REMOTE OBJECT YET, THIS
    CODE IS LEFT ON THE SHELF.  WHEN CLASSINFO IS FIXED, USE THIS CODE:

    We create an identifing name for the class from the classname + unique string
  sidl_ClassInfo clsinfo = sidl_BaseClass_getClassInfo(instance);
  char * clsName = sidl_ClassInfo_getName(clsinfo);
  char * instName = sidl_String_concat2(clsName,next_string(counter));
  sidl_String_free(clsName);
  sidl_ClassInfo_deleteRef(clsinfo);

  hashtable_insert(hshtbl, (void*)instName, (void*)instance);
  return sidl_String_strdup(instName);

  UNTIL THEN, WE USE THIS CODE:
  */
  next_string(counter);
  hashtable_insert(hshtbl, (void*)counter, (void*)instance);
  return sidl_String_strdup(counter);

  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.registerInstance) */
#line 244 "sidl_rmi_InstanceRegistry_Impl.c"
}

/*
 * returns a handle to the class based on the unique string
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_getInstance"

#ifdef __cplusplus
extern "C"
#endif
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_getInstance(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex)
{
#line 252 "../../../babel/runtime/sidl/sidl_rmi_InstanceRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.getInstance) */
  sidl_BaseClass bc = 0;
  /*
  struct sidl_rmi_InstanceRegistry__data *dptr =
    sidl_rmi_InstanceRegistry__get_data(self);
  if(dptr) {
  */
  bc = (sidl_BaseClass) hashtable_search(hshtbl, (void*)instanceID);
  if(bc == 0)
    return 0;
  sidl_BaseClass_addRef(bc);
  return bc;
  /*
  }
  */

  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.getInstance) */
#line 280 "sidl_rmi_InstanceRegistry_Impl.c"
}

/*
 * returns a handle to the class based on the unique string
 * and removes the instance from the table.  
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_InstanceRegistry_removeInstance"

#ifdef __cplusplus
extern "C"
#endif
sidl_BaseClass
impl_sidl_rmi_InstanceRegistry_removeInstance(
  /* in */ const char* instanceID,
  /* out */ sidl_BaseInterface *_ex)
{
#line 287 "../../../babel/runtime/sidl/sidl_rmi_InstanceRegistry_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.InstanceRegistry.removeInstance) */
  sidl_BaseClass bc = 0;
  bc = (sidl_BaseClass) hashtable_remove(hshtbl, (void*)instanceID);
  if(bc == 0)
    return 0;
  return bc;

  /* DO-NOT-DELETE splicer.end(sidl.rmi.InstanceRegistry.removeInstance) */
#line 308 "sidl_rmi_InstanceRegistry_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_rmi_InstanceRegistry__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_InstanceRegistry(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_InstanceRegistry__connect(url, _ex);
}
char * impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_InstanceRegistry(struct 
  sidl_rmi_InstanceRegistry__object* obj) {
  return sidl_rmi_InstanceRegistry__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidl_rmi_InstanceRegistry_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_rmi_InstanceRegistry_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
