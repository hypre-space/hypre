/*
 * File:          sidlx_io_TxtIStream_IOR.h
 * Symbol:        sidlx.io.TxtIStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for sidlx.io.TxtIStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_sidlx_io_TxtIStream_IOR_h
#define included_sidlx_io_TxtIStream_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif
#ifndef included_sidlx_io_IStream_IOR_h
#include "sidlx_io_IStream_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.io.TxtIStream" (version 0.1)
 * 
 * Simple text-based input stream 
 * some datatypes (e.g. strings, arrays, etc require special formatting)
 * undefined behavior with non-whitespace separated fields.
 */

struct sidlx_io_TxtIStream__array;
struct sidlx_io_TxtIStream__object;

extern struct sidlx_io_TxtIStream__object*
sidlx_io_TxtIStream__new(void);

extern void sidlx_io_TxtIStream__init(
  struct sidlx_io_TxtIStream__object* self);
extern void sidlx_io_TxtIStream__fini(
  struct sidlx_io_TxtIStream__object* self);
extern void sidlx_io_TxtIStream__IOR_version(int32_t *major, int32_t *minor);

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

struct sidlx_io_TxtIStream__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in sidlx.io.IStream-v0.1 */
  sidl_bool (*f_atEnd)(
    /* in */ struct sidlx_io_TxtIStream__object* self);
  int32_t (*f_read)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* in */ int32_t nbytes,
    /* out array<char,row-major> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readline)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out array<char,row-major> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getBool)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ sidl_bool* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getChar)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ char* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getInt)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ int32_t* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getLong)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ int64_t* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getFloat)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ float* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getDouble)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ double* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getFcomplex)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ struct sidl_fcomplex* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getDcomplex)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ struct sidl_dcomplex* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getString)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* out */ char** item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.io.TxtIStream-v0.1 */
  void (*f_setFD)(
    /* in */ struct sidlx_io_TxtIStream__object* self,
    /* in */ int32_t fd);
};

/*
 * Define the class object structure.
 */

struct sidlx_io_TxtIStream__object {
  struct sidl_BaseClass__object    d_sidl_baseclass;
  struct sidlx_io_IStream__object  d_sidlx_io_istream;
  struct sidlx_io_TxtIStream__epv* d_epv;
  void*                            d_data;
};

struct sidlx_io_TxtIStream__external {
  struct sidlx_io_TxtIStream__object*
  (*createObject)(void);

  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_io_TxtIStream__external*
sidlx_io_TxtIStream__externals(void);

struct sidlx_io_IOException__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidlx_io_IOException(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtIStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj); 

struct sidl_ClassInfo__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtIStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct sidlx_io_TxtIStream__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidlx_io_TxtIStream(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtIStream_fgetURL_sidlx_io_TxtIStream(struct 
  sidlx_io_TxtIStream__object* obj); 

struct sidl_BaseInterface__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtIStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtIStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

struct sidlx_io_IStream__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidlx_io_IStream(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_sidlx_io_TxtIStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
