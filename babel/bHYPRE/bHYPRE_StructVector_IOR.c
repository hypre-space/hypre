/*
 * File:          bHYPRE_StructVector_IOR.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "bHYPRE_StructVector_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE_StructVector__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_StructVector__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_StructVector__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_StructVector__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variable to make sure _load called no more than once
 */

static int s_load_called = 0;
/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;
static int s_static_initialized = 0;

static struct bHYPRE_StructVector__epv  s_new_epv__bhypre_structvector;
static struct bHYPRE_StructVector__sepv s_stc_epv__bhypre_structvector;

static struct bHYPRE_MatrixVectorView__epv s_new_epv__bhypre_matrixvectorview;

static struct bHYPRE_ProblemDefinition__epv s_new_epv__bhypre_problemdefinition;

static struct bHYPRE_StructVectorView__epv s_new_epv__bhypre_structvectorview;

static struct bHYPRE_Vector__epv s_new_epv__bhypre_vector;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void bHYPRE_StructVector__set_epv(
  struct bHYPRE_StructVector__epv* epv);
extern void bHYPRE_StructVector__set_sepv(
  struct bHYPRE_StructVector__sepv* sepv);
extern void bHYPRE_StructVector__call_load(void);
#ifdef __cplusplus
}
#endif

static void
bHYPRE_StructVector_addRef__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_addRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_deleteRef__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_deleteRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_isSame__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseInterface__object* iobj;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_isSame)(
    self,
    iobj);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_queryInt__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_BaseInterface__object* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_queryInt)(
    self,
    name);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_isType__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_isType)(
    self,
    name);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_getClassInfo__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_SetCommunicator__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* mpi_comm_str= NULL;
  struct bHYPRE_MPICommunicator__object* mpi_comm= NULL;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "mpi_comm", &mpi_comm_str, _ex2);
  mpi_comm = 
    skel_bHYPRE_StructVector_fconnect_bHYPRE_MPICommunicator(mpi_comm_str,
    _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_Initialize__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Initialize)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_Assemble__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Assemble)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_SetGrid__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* grid_str= NULL;
  struct bHYPRE_StructGrid__object* grid= NULL;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "grid", &grid_str, _ex2);
  grid = skel_bHYPRE_StructVector_fconnect_bHYPRE_StructGrid(grid_str, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetGrid)(
    self,
    grid);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_SetNumGhost__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* num_ghost;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_SetNumGhost)(
    self,
    num_ghost);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_SetValue__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* grid_index;
  double value;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackDouble( inArgs, "value", &value, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_SetValue)(
    self,
    grid_index,
    value);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_SetBoxValues__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_int__array* ilower;
  struct sidl_int__array* iupper;
  struct sidl_double__array* values;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_SetBoxValues)(
    self,
    ilower,
    iupper,
    values);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_Clear__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Clear)(
    self);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_Copy__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* x;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Copy)(
    self,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_Clone__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* x_tmp;
  struct bHYPRE_Vector__object** x= &x_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Clone)(
    self,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_Scale__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  double a;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackDouble( inArgs, "a", &a, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_Scale)(
    self,
    a);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
bHYPRE_StructVector_Dot__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct bHYPRE_Vector__object* x;
  double d_tmp;
  double* d= &d_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_Dot)(
    self,
    x,
    d);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packDouble( outArgs, "d", *d, _ex2);

}

static void
bHYPRE_StructVector_Axpy__exec(
        struct bHYPRE_StructVector__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  double a;
  struct bHYPRE_Vector__object* x;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackDouble( inArgs, "a", &a, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_Axpy)(
    self,
    a,
    x);

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void ior_bHYPRE_StructVector__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    bHYPRE_StructVector__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_bHYPRE_StructVector__cast(
  struct bHYPRE_StructVector__object* self,
  const char* name)
{
  void* cast = NULL;

  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.StructVector")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.MatrixVectorView")) {
    cast = (void*) &s0->d_bhypre_matrixvectorview;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "bHYPRE.StructVectorView")) {
    cast = (void*) &s0->d_bhypre_structvectorview;
  } else if (!strcmp(name, "bHYPRE.Vector")) {
    cast = (void*) &s0->d_bhypre_vector;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_bHYPRE_StructVector__delete(
  struct bHYPRE_StructVector__object* self)
{
  bHYPRE_StructVector__fini(self);
  memset((void*)self, 0, sizeof(struct bHYPRE_StructVector__object));
  free((void*) self);
}

static char*
ior_bHYPRE_StructVector__getURL(
    struct bHYPRE_StructVector__object* self) {
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  /* TODO: Make this work for local object! */
  return NULL;
}
struct bHYPRE_StructVector__method {
  const char *d_name;
  void (*d_func)(struct bHYPRE_StructVector__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_bHYPRE_StructVector__exec(
    struct bHYPRE_StructVector__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct bHYPRE_StructVector__method  s_methods[] = {
    { "Assemble", bHYPRE_StructVector_Assemble__exec },
    { "Axpy", bHYPRE_StructVector_Axpy__exec },
    { "Clear", bHYPRE_StructVector_Clear__exec },
    { "Clone", bHYPRE_StructVector_Clone__exec },
    { "Copy", bHYPRE_StructVector_Copy__exec },
    { "Dot", bHYPRE_StructVector_Dot__exec },
    { "Initialize", bHYPRE_StructVector_Initialize__exec },
    { "Scale", bHYPRE_StructVector_Scale__exec },
    { "SetBoxValues", bHYPRE_StructVector_SetBoxValues__exec },
    { "SetCommunicator", bHYPRE_StructVector_SetCommunicator__exec },
    { "SetGrid", bHYPRE_StructVector_SetGrid__exec },
    { "SetNumGhost", bHYPRE_StructVector_SetNumGhost__exec },
    { "SetValue", bHYPRE_StructVector_SetValue__exec },
    { "addRef", bHYPRE_StructVector_addRef__exec },
    { "deleteRef", bHYPRE_StructVector_deleteRef__exec },
    { "getClassInfo", bHYPRE_StructVector_getClassInfo__exec },
    { "isSame", bHYPRE_StructVector_isSame__exec },
    { "isType", bHYPRE_StructVector_isType__exec },
    { "queryInt", bHYPRE_StructVector_queryInt__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct bHYPRE_StructVector__method);
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
}
/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void bHYPRE_StructVector__init_epv(
  struct bHYPRE_StructVector__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  struct bHYPRE_StructVector__epv*      epv  = &s_new_epv__bhypre_structvector;
  struct bHYPRE_MatrixVectorView__epv*  e0   = 
    &s_new_epv__bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__epv* e1   = 
    &s_new_epv__bhypre_problemdefinition;
  struct bHYPRE_StructVectorView__epv*  e2   = 
    &s_new_epv__bhypre_structvectorview;
  struct bHYPRE_Vector__epv*            e3   = &s_new_epv__bhypre_vector;
  struct sidl_BaseClass__epv*           e4   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e5   = &s_new_epv__sidl_baseinterface;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_bHYPRE_StructVector__cast;
  epv->f__delete                  = ior_bHYPRE_StructVector__delete;
  epv->f__exec                    = ior_bHYPRE_StructVector__exec;
  epv->f__getURL                  = ior_bHYPRE_StructVector__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    bHYPRE_StructVector__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    bHYPRE_StructVector__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    bHYPRE_StructVector__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct bHYPRE_StructVector__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    bHYPRE_StructVector__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    bHYPRE_StructVector__object*)) s1->d_epv->f_getClassInfo;
  epv->f_SetCommunicator          = NULL;
  epv->f_Initialize               = NULL;
  epv->f_Assemble                 = NULL;
  epv->f_SetGrid                  = NULL;
  epv->f_SetNumGhost              = NULL;
  epv->f_SetValue                 = NULL;
  epv->f_SetBoxValues             = NULL;
  epv->f_Clear                    = NULL;
  epv->f_Copy                     = NULL;
  epv->f_Clone                    = NULL;
  epv->f_Scale                    = NULL;
  epv->f_Dot                      = NULL;
  epv->f_Axpy                     = NULL;

  bHYPRE_StructVector__set_epv(epv);

  e0->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete             = (void (*)(void*)) epv->f__delete;
  e0->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e0->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;

  e1->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete             = (void (*)(void*)) epv->f__delete;
  e1->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e1->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e1->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;

  e2->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete             = (void (*)(void*)) epv->f__delete;
  e2->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_SetCommunicator     = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e2->f_Initialize          = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble            = (int32_t (*)(void*)) epv->f_Assemble;
  e2->f_SetGrid             = (int32_t (*)(void*,
    struct bHYPRE_StructGrid__object*)) epv->f_SetGrid;
  e2->f_SetNumGhost         = (int32_t (*)(void*,
    struct sidl_int__array*)) epv->f_SetNumGhost;
  e2->f_SetValue            = (int32_t (*)(void*,struct sidl_int__array*,
    double)) epv->f_SetValue;
  e2->f_SetBoxValues        = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_double__array*)) epv->f_SetBoxValues;

  e3->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete             = (void (*)(void*)) epv->f__delete;
  e3->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_Clear               = (int32_t (*)(void*)) epv->f_Clear;
  e3->f_Copy                = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*)) epv->f_Copy;
  e3->f_Clone               = (int32_t (*)(void*,
    struct bHYPRE_Vector__object**)) epv->f_Clone;
  e3->f_Scale               = (int32_t (*)(void*,double)) epv->f_Scale;
  e3->f_Dot                 = (int32_t (*)(void*,struct bHYPRE_Vector__object*,
    double*)) epv->f_Dot;
  e3->f_Axpy                = (int32_t (*)(void*,double,
    struct bHYPRE_Vector__object*)) epv->f_Axpy;

  e4->f__cast               = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e4->f__delete             = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e4->f__exec               = (void (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e4->f_addRef              = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_addRef;
  e4->f_deleteRef           = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e4->f_isSame              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt            = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e4->f_isType              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e4->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e5->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e5->f__delete             = (void (*)(void*)) epv->f__delete;
  e5->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e5->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e5->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e5->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e5->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e5->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_method_initialized = 1;
  ior_bHYPRE_StructVector__ensure_load_called();
}

/*
 * SEPV: create the static entry point vector (SEPV).
 */

static void bHYPRE_StructVector__init_sepv(void)
{
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  struct bHYPRE_StructVector__sepv*  s = &s_stc_epv__bhypre_structvector;

  s->f_Create         = NULL;

  bHYPRE_StructVector__set_sepv(s);

  s_static_initialized = 1;
  ior_bHYPRE_StructVector__ensure_load_called();
}

/*
 * STATIC: return pointer to static EPV structure.
 */

struct bHYPRE_StructVector__sepv*
bHYPRE_StructVector__statics(void)
{
  LOCK_STATIC_GLOBALS;
  if (!s_static_initialized) {
    bHYPRE_StructVector__init_sepv();
  }
  UNLOCK_STATIC_GLOBALS;
  return &s_stc_epv__bhypre_structvector;
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* bHYPRE_StructVector__super(void) {
  return s_old_epv__sidl_baseclass;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info)
{
  LOCK_STATIC_GLOBALS;
  if (s_classInfo_init) {
    sidl_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = sidl_ClassInfoI__create();
    s_classInfo = sidl_ClassInfo__cast(impl);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "bHYPRE.StructVector");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info);
  }
UNLOCK_STATIC_GLOBALS;
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct bHYPRE_StructVector__object* self)
{
  if (self) {
    struct sidl_BaseClass__data *data = 
      sidl_BaseClass__get_data(sidl_BaseClass__cast(self));
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo));
    }
  }
}

/*
 * NEW: allocate object and initialize it.
 */

struct bHYPRE_StructVector__object*
bHYPRE_StructVector__new(void)
{
  struct bHYPRE_StructVector__object* self =
    (struct bHYPRE_StructVector__object*) malloc(
      sizeof(struct bHYPRE_StructVector__object));
  bHYPRE_StructVector__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void bHYPRE_StructVector__init(
  struct bHYPRE_StructVector__object* self)
{
  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    bHYPRE_StructVector__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv  = &s_new_epv__bhypre_matrixvectorview;
  s0->d_bhypre_problemdefinition.d_epv = &s_new_epv__bhypre_problemdefinition;
  s0->d_bhypre_structvectorview.d_epv  = &s_new_epv__bhypre_structvectorview;
  s0->d_bhypre_vector.d_epv            = &s_new_epv__bhypre_vector;
  s0->d_epv                            = &s_new_epv__bhypre_structvector;

  s0->d_bhypre_matrixvectorview.d_object = self;

  s0->d_bhypre_problemdefinition.d_object = self;

  s0->d_bhypre_structvectorview.d_object = self;

  s0->d_bhypre_vector.d_object = self;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void bHYPRE_StructVector__fini(
  struct bHYPRE_StructVector__object* self)
{
  struct bHYPRE_StructVector__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
bHYPRE_StructVector__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct bHYPRE_StructVector__external
s_externalEntryPoints = {
  bHYPRE_StructVector__new,
  bHYPRE_StructVector__statics,
  bHYPRE_StructVector__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_StructVector__external*
bHYPRE_StructVector__externals(void)
{
  return &s_externalEntryPoints;
}

