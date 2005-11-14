/*
 * File:          sidlx_io_TxtOStream_IOR.h
 * Symbol:        sidlx.io.TxtOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.io.TxtOStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_io_TxtOStream_IOR_h
#define included_sidlx_io_TxtOStream_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidlx_io_OStream_IOR_h
#include "sidlx_io_OStream_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.io.TxtOStream" (version 0.1)
 * 
 * Simple text-based output stream appends spaces
 */

struct sidlx_io_TxtOStream__array;
struct sidlx_io_TxtOStream__object;

extern struct sidlx_io_TxtOStream__object*
sidlx_io_TxtOStream__new(void);

extern void sidlx_io_TxtOStream__init(
  struct sidlx_io_TxtOStream__object* self);
extern void sidlx_io_TxtOStream__fini(
  struct sidlx_io_TxtOStream__object* self);
extern void sidlx_io_TxtOStream__IOR_version(int32_t *major, int32_t *minor);

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
struct sidlx_io_IOException__array;
struct sidlx_io_IOException__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_io_TxtOStream__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidlx.io.OStream-v0.1 */
  void (*f_flush)(
    /* in */ struct sidlx_io_TxtOStream__object* self);
  int32_t (*f_write)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in array<char,row-major> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putBool)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ sidl_bool item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putChar)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ char item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putInt)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ int32_t item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putLong)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ int64_t item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putFloat)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ float item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putDouble)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ double item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putFcomplex)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ struct sidl_fcomplex item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putDcomplex)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ struct sidl_dcomplex item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putString)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ const char* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.io.TxtOStream-v0.1 */
  void (*f_setFD)(
    /* in */ struct sidlx_io_TxtOStream__object* self,
    /* in */ int32_t fd);
};

/*
 * Define the class object structure.
 */

struct sidlx_io_TxtOStream__object {
  struct sidl_BaseClass__object    d_sidl_baseclass;
  struct sidlx_io_OStream__object  d_sidlx_io_ostream;
  struct sidlx_io_TxtOStream__epv* d_epv;
  void*                            d_data;
};

struct sidlx_io_TxtOStream__external {
  struct sidlx_io_TxtOStream__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_io_TxtOStream__external*
sidlx_io_TxtOStream__externals(void);

struct sidlx_io_TxtOStream__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidlx_io_TxtOStream(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtOStream_fgetURL_sidlx_io_TxtOStream(struct 
  sidlx_io_TxtOStream__object* obj); 

struct sidlx_io_IOException__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidlx_io_IOException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidlx_io_OStream__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidlx_io_OStream(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
