/*
 * File:          sidl_rmi_ProtocolFactory_IOR.h
 * Symbol:        sidl.rmi.ProtocolFactory-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Release:       $Name: V2-4-0b $
 * Revision:      @(#) $Id: sidl_rmi_ProtocolFactory_IOR.h,v 1.6 2007/09/27 19:35:47 painter Exp $
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
 */

#ifndef included_sidl_rmi_ProtocolFactory_IOR_h
#define included_sidl_rmi_ProtocolFactory_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
struct sidl_rmi_InstanceHandle__object;
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidl.rmi.ProtocolFactory" (version 0.9.15)
 * 
 *  
 * This singleton class keeps a table of string prefixes
 * (e.g. "babel" or "proteus") to protocol implementations.  The
 * intent is to parse a URL (e.g. "babel://server:port/class") and
 * create classes that implement
 * <code>sidl.rmi.InstanceHandle</code>.
 */

struct sidl_rmi_ProtocolFactory__array;
struct sidl_rmi_ProtocolFactory__object;
struct sidl_rmi_ProtocolFactory__sepv;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseException__array;
struct sidl_BaseException__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_RuntimeException__array;
struct sidl_RuntimeException__object;
struct sidl_io_Serializable__array;
struct sidl_io_Serializable__object;
struct sidl_rmi_Call__array;
struct sidl_rmi_Call__object;
struct sidl_rmi_InstanceHandle__array;
struct sidl_rmi_InstanceHandle__object;
struct sidl_rmi_Return__array;
struct sidl_rmi_Return__object;

/*
 * Declare the static method entry point vector.
 */

struct sidl_rmi_ProtocolFactory__sepv {
  /* Implicit builtin methods */
  /* 0 */
  /* 1 */
  /* 2 */
  /* 3 */
  /* 4 */
  /* 5 */
  /* 6 */
  void (*f__set_hooks_static)(
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  /* 8 */
  /* 9 */
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidl.rmi.ProtocolFactory-v0.9.15 */
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
    /* in */ sidl_bool ar,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_io_Serializable__object* (*f_unserializeInstance)(
    /* in */ const char* url,
    /* out */ struct sidl_BaseInterface__object* *_ex);
};

/*
 * Declare the method entry point vector.
 */

struct sidl_rmi_ProtocolFactory__epv {
  /* Implicit builtin methods */
  /* 0 */
  void* (*f__cast)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 1 */
  void (*f__delete)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 2 */
  void (*f__exec)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_rmi_Call__object* inArgs,
    /* in */ struct sidl_rmi_Return__object* outArgs,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 3 */
  char* (*f__getURL)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 4 */
  void (*f__raddRef)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 5 */
  sidl_bool (*f__isRemote)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 6 */
  void (*f__set_hooks)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 7 */
  void (*f__ctor)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 8 */
  void (*f__ctor2)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ void* private_data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 9 */
  void (*f__dtor)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* 10 */
  /* 11 */
  /* 12 */
  /* 13 */
  /* Methods introduced in sidl.BaseInterface-v0.9.15 */
  void (*f_addRef)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_deleteRef)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isSame)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  sidl_bool (*f_isType)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidl_rmi_ProtocolFactory__object* self,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidl.BaseClass-v0.9.15 */
  /* Methods introduced in sidl.rmi.ProtocolFactory-v0.9.15 */
};

/*
 * Define the controls structure.
 */


struct sidl_rmi_ProtocolFactory__controls {
  int     use_hooks;
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
  (*createObject)(void* ddata, struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ProtocolFactory__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
  int d_ior_major_version;
  int d_ior_minor_version;
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_rmi_ProtocolFactory__external*
sidl_rmi_ProtocolFactory__externals(void);

extern struct sidl_rmi_ProtocolFactory__object*
sidl_rmi_ProtocolFactory__new(void* ddata,struct sidl_BaseInterface__object ** 
  _ex);

extern struct sidl_rmi_ProtocolFactory__sepv*
sidl_rmi_ProtocolFactory__statics(void);

extern void sidl_rmi_ProtocolFactory__init(
  struct sidl_rmi_ProtocolFactory__object* self, void* ddata, struct 
    sidl_BaseInterface__object ** _ex);
extern void sidl_rmi_ProtocolFactory__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_rmi_ProtocolFactory__epv **s_arg_epv__sidl_rmi_protocolfactory,
    struct sidl_rmi_ProtocolFactory__epv 
    **s_arg_epv_hooks__sidl_rmi_protocolfactory);
  extern void sidl_rmi_ProtocolFactory__fini(
    struct sidl_rmi_ProtocolFactory__object* self, struct 
      sidl_BaseInterface__object ** _ex);
  extern void sidl_rmi_ProtocolFactory__IOR_version(int32_t *major, int32_t 
    *minor);

  struct sidl_BaseClass__object* 
    skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseClass__object* 
    skel_sidl_rmi_ProtocolFactory_fcast_sidl_BaseClass(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_BaseInterface__object* 
    skel_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_BaseInterface__object* 
    skel_sidl_rmi_ProtocolFactory_fcast_sidl_BaseInterface(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_ClassInfo__object* 
    skel_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(const char* url, 
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_ClassInfo__object* 
    skel_sidl_rmi_ProtocolFactory_fcast_sidl_ClassInfo(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_RuntimeException__object* 
    skel_sidl_rmi_ProtocolFactory_fconnect_sidl_RuntimeException(const char* 
    url, sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_RuntimeException__object* 
    skel_sidl_rmi_ProtocolFactory_fcast_sidl_RuntimeException(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_io_Serializable__object* 
    skel_sidl_rmi_ProtocolFactory_fconnect_sidl_io_Serializable(const char* url,
    sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_io_Serializable__object* 
    skel_sidl_rmi_ProtocolFactory_fcast_sidl_io_Serializable(void *bi, struct 
    sidl_BaseInterface__object **_ex);

  struct sidl_rmi_InstanceHandle__object* 
    skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(const char* 
    url, sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_InstanceHandle__object* 
    skel_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_InstanceHandle(void *bi, 
    struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ProtocolFactory__object* 
    skel_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(const char* 
    url, sidl_bool ar, struct sidl_BaseInterface__object **_ex);
  struct sidl_rmi_ProtocolFactory__object* 
    skel_sidl_rmi_ProtocolFactory_fcast_sidl_rmi_ProtocolFactory(void *bi, 
    struct sidl_BaseInterface__object **_ex);

  struct sidl_rmi_ProtocolFactory__remote{
    int d_refcount;
    struct sidl_rmi_InstanceHandle__object *d_ih;
  };

#ifdef __cplusplus
  }
#endif
#endif
