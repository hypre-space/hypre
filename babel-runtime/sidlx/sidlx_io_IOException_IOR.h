/*
 * File:          sidlx_io_IOException_IOR.h
 * Symbol:        sidlx.io.IOException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for sidlx.io.IOException
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_sidlx_io_IOException_IOR_h
#define included_sidlx_io_IOException_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_IOR_h
#include "sidl_SIDLException_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.io.IOException" (version 0.1)
 */

struct sidlx_io_IOException__array;
struct sidlx_io_IOException__object;

extern struct sidlx_io_IOException__object*
sidlx_io_IOException__new(void);

extern void sidlx_io_IOException__init(
  struct sidlx_io_IOException__object* self);
extern void sidlx_io_IOException__fini(
  struct sidlx_io_IOException__object* self);
extern void sidlx_io_IOException__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_io_IOException__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_io_IOException__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_io_IOException__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_io_IOException__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_io_IOException__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_io_IOException__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_io_IOException__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_io_IOException__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidl.BaseException-v0.9.3 */
  char* (*f_getNote)(
    /* in */ struct sidlx_io_IOException__object* self);
  void (*f_setNote)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ const char* message);
  char* (*f_getTrace)(
    /* in */ struct sidlx_io_IOException__object* self);
  void (*f_addLine)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ const char* traceline);
  void (*f_add)(
    /* in */ struct sidlx_io_IOException__object* self,
    /* in */ const char* filename,
    /* in */ int32_t lineno,
    /* in */ const char* methodname);
  /* Methods introduced in sidl.SIDLException-v0.9.3 */
  /* Methods introduced in sidlx.io.IOException-v0.1 */
};

/*
 * Define the class object structure.
 */

struct sidlx_io_IOException__object {
  struct sidl_SIDLException__object d_sidl_sidlexception;
  struct sidlx_io_IOException__epv* d_epv;
  void*                             d_data;
};

struct sidlx_io_IOException__external {
  struct sidlx_io_IOException__object*
  (*createObject)(void);

  struct sidl_SIDLException__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_io_IOException__external*
sidlx_io_IOException__externals(void);

struct sidl_SIDLException__object* 
  skel_sidlx_io_IOException_fconnect_sidl_SIDLException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_IOException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj); 

struct sidlx_io_IOException__object* 
  skel_sidlx_io_IOException_fconnect_sidlx_io_IOException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_IOException_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_io_IOException_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_IOException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_io_IOException_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_IOException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseException__object* 
  skel_sidlx_io_IOException_fconnect_sidl_BaseException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_IOException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_io_IOException_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_IOException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
