/*
 * File:          sidl_rmi_ProtocolFactory_IOR.h
 * Symbol:        sidl.rmi.ProtocolFactory-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.rmi.ProtocolFactory
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

#ifndef included_sidl_rmi_ProtocolFactory_IOR_h
#define included_sidl_rmi_ProtocolFactory_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.rmi.ProtocolFactory" (version 0.9.3)
 * 
 * This singleton class keeps a table of string prefixes (e.g. "babel" or "proteus")
 * to protocol implementations.  The intent is to parse a URL (e.g. "babel://server:port/class")
 * and create classes that implement <code>sidl.rmi.InstanceHandle</code>.
 */

struct sidl_rmi_ProtocolFactory__array;
struct sidl_rmi_ProtocolFactory__object;
struct sidl_rmi_ProtocolFactory__sepv;

extern struct sidl_rmi_ProtocolFactory__object*
sidl_rmi_ProtocolFactory__new(void);

extern struct sidl_rmi_ProtocolFactory__sepv*
sidl_rmi_ProtocolFactory__statics(void);

extern void sidl_rmi_ProtocolFactory__init(
  struct sidl_rmi_ProtocolFactory__object* self);
extern void sidl_rmi_ProtocolFactory__fini(
  struct sidl_rmi_ProtocolFactory__object* self);
extern void sidl_rmi_ProtocolFactory__IOR_version(int32_t *major,
  int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;
struct sidl_rmi_InstanceHandle__array;
struct sidl_rmi_InstanceHandle__object;
struct sidl_rmi_NetworkException__array;
struct sidl_rmi_NetworkException__object;

/*
 * Declare the static method entry point vector.
 */

struct sidl_rmi_ProtocolFactory__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.rmi.ProtocolFactory-v0.9.3 */
  sidl_bool (*f_addProtocol)(
    /* in */ const char* prefix,
    /* in */ const char* typeName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  char* (*f_getProtocol)(
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_deleteProtocol)(
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_rmi_InstanceHandle__object* (*f_createInstance)(
    /* in */ const char* url,
    /* in */ const char* typeName,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_rmi_InstanceHandle__object* (*f_connectInstance)(
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct sidl_rmi_ProtocolFactory__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self);
  void (*f__exec)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self);
  void (*f__ctor)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self);
  void (*f__dtor)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.rmi.ProtocolFactory-v0.9.3 */
};

/*
 * Define the class object structure.
 */

struct sidl_rmi_ProtocolFactory__object {
  struct sidl_BaseClass__object         d_sidl_baseclass;
  struct sidl_rmi_ProtocolFactory__epv* d_epv;
  void*                                 d_data;
};

struct sidl_rmi_ProtocolFactory__external {
  struct sidl_rmi_ProtocolFactory__object*
  (*createObject)(void);

  struct sidl_rmi_ProtocolFactory__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_rmi_ProtocolFactory__external*
sidl_rmi_ProtocolFactory__externals(void);

struct sidl_rmi_InstanceHandle__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_rmi_NetworkException__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_rmi_ProtocolFactory__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(struct 
  sidl_rmi_ProtocolFactory__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
