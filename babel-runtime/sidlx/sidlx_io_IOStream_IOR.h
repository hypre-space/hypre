/*
 * File:          sidlx_io_IOStream_IOR.h
 * Symbol:        sidlx.io.IOStream-v0.1
 * Symbol Type:   interface
 * Babel Version: 0.10.10
 * Description:   Intermediate Object Representation for sidlx.io.IOStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#ifndef included_sidlx_io_IOStream_IOR_h
#define included_sidlx_io_IOStream_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.io.IOStream" (version 0.1)
 */

struct sidlx_io_IOStream__array;
struct sidlx_io_IOStream__object;

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

struct sidlx_io_IOStream__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ void* self);
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ void* self);
  void (*f_deleteRef)(
    /* in */ void* self);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ void* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self);
  /* Methods introduced in sidlx.io.IStream-v0.1 */
  sidl_bool (*f_atEnd)(
    /* in */ void* self);
  int32_t (*f_read)(
    /* in */ void* self,
    /* in */ int32_t nbytes,
    /* out array<char,row-major> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readline)(
    /* in */ void* self,
    /* out array<char,row-major> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getBool)(
    /* in */ void* self,
    /* out */ sidl_bool* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getChar)(
    /* in */ void* self,
    /* out */ char* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getInt)(
    /* in */ void* self,
    /* out */ int32_t* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getLong)(
    /* in */ void* self,
    /* out */ int64_t* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getFloat)(
    /* in */ void* self,
    /* out */ float* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getDouble)(
    /* in */ void* self,
    /* out */ double* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getFcomplex)(
    /* in */ void* self,
    /* out */ struct sidl_fcomplex* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getDcomplex)(
    /* in */ void* self,
    /* out */ struct sidl_dcomplex* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getString)(
    /* in */ void* self,
    /* out */ char** item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.io.OStream-v0.1 */
  void (*f_flush)(
    /* in */ void* self);
  int32_t (*f_write)(
    /* in */ void* self,
    /* in array<char,row-major> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putBool)(
    /* in */ void* self,
    /* in */ sidl_bool item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putChar)(
    /* in */ void* self,
    /* in */ char item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putInt)(
    /* in */ void* self,
    /* in */ int32_t item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putLong)(
    /* in */ void* self,
    /* in */ int64_t item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putFloat)(
    /* in */ void* self,
    /* in */ float item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putDouble)(
    /* in */ void* self,
    /* in */ double item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putFcomplex)(
    /* in */ void* self,
    /* in */ struct sidl_fcomplex item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putDcomplex)(
    /* in */ void* self,
    /* in */ struct sidl_dcomplex item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putString)(
    /* in */ void* self,
    /* in */ const char* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.io.IOStream-v0.1 */
};

/*
 * Define the interface object structure.
 */

struct sidlx_io_IOStream__object {
  struct sidlx_io_IOStream__epv* d_epv;
  void*                          d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif
#ifndef included_sidlx_io_IOStream_IOR_h
#include "sidlx_io_IOStream_IOR.h"
#endif
#ifndef included_sidlx_io_IStream_IOR_h
#include "sidlx_io_IStream_IOR.h"
#endif
#ifndef included_sidlx_io_OStream_IOR_h
#include "sidlx_io_OStream_IOR.h"
#endif

/*
 * Symbol "sidlx.io._IOStream" (version 1.0)
 */

struct sidlx_io__IOStream__array;
struct sidlx_io__IOStream__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_io__IOStream__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_io__IOStream__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_io__IOStream__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_io__IOStream__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_io__IOStream__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_io__IOStream__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_io__IOStream__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_io__IOStream__object* self);
  /* Methods introduced in sidlx.io.IStream-v0.1 */
  sidl_bool (*f_atEnd)(
    /* in */ struct sidlx_io__IOStream__object* self);
  int32_t (*f_read)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ int32_t nbytes,
    /* out array<char,row-major> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  int32_t (*f_readline)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out array<char,row-major> */ struct sidl_char__array** data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getBool)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ sidl_bool* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getChar)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ char* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getInt)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ int32_t* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getLong)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ int64_t* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getFloat)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ float* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getDouble)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ double* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getFcomplex)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ struct sidl_fcomplex* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getDcomplex)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ struct sidl_dcomplex* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_getString)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* out */ char** item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.io.OStream-v0.1 */
  void (*f_flush)(
    /* in */ struct sidlx_io__IOStream__object* self);
  int32_t (*f_write)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in array<char,row-major> */ struct sidl_char__array* data,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putBool)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ sidl_bool item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putChar)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ char item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putInt)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ int32_t item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putLong)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ int64_t item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putFloat)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ float item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putDouble)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ double item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putFcomplex)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ struct sidl_fcomplex item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putDcomplex)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ struct sidl_dcomplex item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  void (*f_putString)(
    /* in */ struct sidlx_io__IOStream__object* self,
    /* in */ const char* item,
    /* out */ struct sidl_BaseInterface__object* *_ex);
  /* Methods introduced in sidlx.io.IOStream-v0.1 */
  /* Methods introduced in sidlx.io._IOStream-v1.0 */
};

/*
 * Define the class object structure.
 */

struct sidlx_io__IOStream__object {
  struct sidl_BaseInterface__object d_sidl_baseinterface;
  struct sidlx_io_IOStream__object  d_sidlx_io_iostream;
  struct sidlx_io_IStream__object   d_sidlx_io_istream;
  struct sidlx_io_OStream__object   d_sidlx_io_ostream;
  struct sidlx_io__IOStream__epv*   d_epv;
  void*                             d_data;
};


#ifdef __cplusplus
}
#endif
#endif
