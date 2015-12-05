/*
 * File:          bHYPRE_StructVector_fStub.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

/*
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */

#ifndef included_bHYPRE_StructVector_fStub_h
#include "bHYPRE_StructVector_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
#ifndef included_sidlf90array_h
#include "sidlf90array.h"
#endif
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_Exception_h
#include "sidl_Exception.h"
#endif
#include <stdio.h>
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "bHYPRE_StructVector_IOR.h"
#include "bHYPRE_StructVector_fAbbrev.h"
#include "bHYPRE_MPICommunicator_IOR.h"
#include "bHYPRE_StructGrid_IOR.h"
#include "bHYPRE_Vector_IOR.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_bHYPRE_MPICommunicator_fStub_h
#include "bHYPRE_MPICommunicator_fStub.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_fStub_h
#include "bHYPRE_MatrixVectorView_fStub.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_fStub_h
#include "bHYPRE_ProblemDefinition_fStub.h"
#endif
#ifndef included_bHYPRE_StructGrid_fStub_h
#include "bHYPRE_StructGrid_fStub.h"
#endif
#ifndef included_bHYPRE_StructVector_fStub_h
#include "bHYPRE_StructVector_fStub.h"
#endif
#ifndef included_bHYPRE_StructVectorView_fStub_h
#include "bHYPRE_StructVectorView_fStub.h"
#endif
#ifndef included_bHYPRE_Vector_fStub_h
#include "bHYPRE_Vector_fStub.h"
#endif
#ifndef included_sidl_BaseClass_fStub_h
#include "sidl_BaseClass_fStub.h"
#endif
#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

static struct bHYPRE_StructVector__object* 
  bHYPRE_StructVector__remoteCreate(const char* url, sidl_BaseInterface *_ex);
static struct bHYPRE_StructVector__object* 
  bHYPRE_StructVector__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct bHYPRE_StructVector__object* 
  bHYPRE_StructVector__IHConnect(struct sidl_rmi_InstanceHandle__object 
  *instance, struct sidl_BaseInterface__object **_ex);
/*
 * Return pointer to internal IOR functions.
 */

static const struct bHYPRE_StructVector__external* _getIOR(void)
{
  static const struct bHYPRE_StructVector__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = bHYPRE_StructVector__externals();
#else
    _ior = (struct 
      bHYPRE_StructVector__external*)sidl_dynamicLoadIOR("bHYPRE.StructVector",
      "bHYPRE_StructVector__externals") ;
    sidl_checkIORVersion("bHYPRE.StructVector", _ior->d_ior_major_version,
      _ior->d_ior_minor_version, 0, 10);
#endif
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct bHYPRE_StructVector__sepv* _getSEPV(void)
{
  static const struct bHYPRE_StructVector__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran90Symbol(bhypre_structvector_newlocal_m,BHYPRE_STRUCTVECTOR_NEWLOCAL_M,bHYPRE_StructVector_newLocal_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object *_ior_exception = NULL;
  *self = (ptrdiff_t) (*(_getIOR()->createObject))(NULL,&_ior_exception);
  *exception = (ptrdiff_t)_ior_exception;
  if (_ior_exception) *self = 0;
}

/*
 * Remote Constructor for the class.
 */

void
SIDLFortran90Symbol(bhypre_structvector_newremote_m,BHYPRE_STRUCTVECTOR_NEWREMOTE_M,bHYPRE_StructVector_newRemote_m)
(
  int64_t *self,
  SIDL_F90_String url
  SIDL_F90_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(url)
)
{
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F90_STR(url),
      SIDL_F90_STR_LEN(url));
  _proxy_self = bHYPRE_StructVector__remoteCreate(_proxy_url,
    &_proxy_exception);
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *self = (ptrdiff_t)_proxy_self;
  }
  free((void *)_proxy_url);
}
/*
 * Data Wrapper for the class.
 */

void
SIDLFortran90Symbol(bhypre_structvector_wrapobj_m,BHYPRE_STRUCTVECTOR_WRAPOBJ_M,bHYPRE_StructVector_wrapObj_m)
(
  void ** private_data,
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object *_ior_exception = NULL;
  void* _proxy_private_data = NULL;
  _proxy_private_data = malloc(SIDL_F90_POINTER_SIZE);
  if (_proxy_private_data) { 
    memcpy(_proxy_private_data, private_data, SIDL_F90_POINTER_SIZE);
  } else {
    fputs("ctor2 failed in a malloc call!", stderr);
  }
  *self = (ptrdiff_t) (*(_getIOR()->createObject))(_proxy_private_data,
    &_ior_exception);
  *exception = (ptrdiff_t)_ior_exception;
  if (_ior_exception) *self = 0;
}

/*
 * Remote Connector for the class.
 */

void
SIDLFortran90Symbol(bhypre_structvector_rconnect_m,BHYPRE_STRUCTVECTOR_RCONNECT_M,bHYPRE_StructVector_rConnect_m)
(
  int64_t *self,
  SIDL_F90_String url
  SIDL_F90_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(url)
)
{
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F90_STR(url),
      SIDL_F90_STR_LEN(url));
  _proxy_self = bHYPRE_StructVector__remoteConnect(_proxy_url, 1,
    &_proxy_exception);
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *self = (ptrdiff_t)_proxy_self;
  }
  free((void *)_proxy_url);
}
/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran90Symbol(bhypre_structvector__cast_m,BHYPRE_STRUCTVECTOR__CAST_M,bHYPRE_StructVector__cast_m)
(
  int64_t *ref,
  int64_t *retval,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  struct sidl_BaseInterface__object *proxy_exception;

  *retval = 0;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.StructVector",
      (void*)bHYPRE_StructVector__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.StructVector", &proxy_exception);
  } else {
    *retval = 0;
    proxy_exception = 0;
  }
  EXIT:
  *exception = (ptrdiff_t)proxy_exception;
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran90Symbol(bhypre_structvector__cast2_m,BHYPRE_STRUCTVECTOR__CAST2_M,bHYPRE_StructVector__cast2_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
  free((void *)_proxy_name);
}


/*
 * Select and execute a method by name
 */

void
SIDLFortran90Symbol(bhypre_structvector__exec_m,BHYPRE_STRUCTVECTOR__EXEC_M,bHYPRE_StructVector__exec_m)
(
  int64_t *self,
  SIDL_F90_String methodName
  SIDL_F90_STR_NEAR_LEN_DECL(methodName),
  int64_t *inArgs,
  int64_t *outArgs,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(methodName)
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_methodName =
    sidl_copy_fortran_str(SIDL_F90_STR(methodName),
      SIDL_F90_STR_LEN(methodName));
  _proxy_inArgs =
    (struct sidl_rmi_Call__object*)
    (ptrdiff_t)(*inArgs);
  _proxy_outArgs =
    (struct sidl_rmi_Return__object*)
    (ptrdiff_t)(*outArgs);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__exec))(
    _proxy_self,
    _proxy_methodName,
    _proxy_inArgs,
    _proxy_outArgs,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_methodName);
}


/*
 * Get the URL of the Implementation of this object (for RMI)
 */

void
SIDLFortran90Symbol(bhypre_structvector__geturl_m,BHYPRE_STRUCTVECTOR__GETURL_M,bHYPRE_StructVector__getURL_m)
(
  int64_t *self,
  SIDL_F90_String retval
  SIDL_F90_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(retval)
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__getURL))(
      _proxy_self,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    sidl_copy_c_str(
      SIDL_F90_STR(retval),
      SIDL_F90_STR_LEN(retval),
      _proxy_retval);
  }
  free((void *)_proxy_retval);
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(bhypre_structvector__isremote_m,BHYPRE_STRUCTVECTOR__ISREMOTE_M,bHYPRE_StructVector__isRemote_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__isRemote))(
      _proxy_self,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran90Symbol(bhypre_structvector__islocal_m,BHYPRE_STRUCTVECTOR__ISLOCAL_M,bHYPRE_StructVector__isLocal_m)
(
  int64_t *self,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    !(*(_epv->f__isRemote))(
      _proxy_self,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}


/*
 * Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran90Symbol(bhypre_structvector__set_hooks_m,BHYPRE_STRUCTVECTOR__SET_HOOKS_M,bHYPRE_StructVector__set_hooks_m)
(
  int64_t *self,
  SIDL_F90_Bool *on,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_on = ((*on == SIDL_F90_TRUE) ? TRUE : FALSE);
  _epv = _proxy_self->d_epv;
  (*(_epv->f__set_hooks))(
    _proxy_self,
    _proxy_on,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}


/*
 * Static Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran90Symbol(bhypre_structvector__set_hooks_static_m,BHYPRE_STRUCTVECTOR__SET_HOOKS_STATIC_M,bHYPRE_StructVector__set_hooks_static_m)
(
  SIDL_F90_Bool *on,
  int64_t *exception
)
{
  const struct bHYPRE_StructVector__sepv *_epv = _getSEPV();
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_on = ((*on == SIDL_F90_TRUE) ? TRUE : FALSE);
  (*(_epv->f__set_hooks_static))(
    _proxy_on,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 *  This function is the preferred way to create a Struct Vector. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_create_m,BHYPRE_STRUCTVECTOR_CREATE_M,bHYPRE_StructVector_Create_m)
(
  int64_t *mpi_comm,
  int64_t *grid,
  int64_t *retval,
  int64_t *exception
)
{
  const struct bHYPRE_StructVector__sepv *_epv = _getSEPV();
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  struct bHYPRE_StructGrid__object* _proxy_grid = NULL;
  struct bHYPRE_StructVector__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _proxy_grid =
    (struct bHYPRE_StructGrid__object*)
    (ptrdiff_t)(*grid);
  _proxy_retval = 
    (*(_epv->f_Create))(
      _proxy_mpi_comm,
      _proxy_grid,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
}

/*
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */

void
SIDLFortran90Symbol(bhypre_structvector_addref_m,BHYPRE_STRUCTVECTOR_ADDREF_M,bHYPRE_StructVector_addRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran90Symbol(bhypre_structvector_deleteref_m,BHYPRE_STRUCTVECTOR_DELETEREF_M,bHYPRE_StructVector_deleteRef_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran90Symbol(bhypre_structvector_issame_m,BHYPRE_STRUCTVECTOR_ISSAME_M,bHYPRE_StructVector_isSame_m)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F90_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self,
      _proxy_iobj,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran90Symbol(bhypre_structvector_istype_m,BHYPRE_STRUCTVECTOR_ISTYPE_M,bHYPRE_StructVector_isType_m)
(
  int64_t *self,
  SIDL_F90_String name
  SIDL_F90_STR_NEAR_LEN_DECL(name),
  SIDL_F90_Bool *retval,
  int64_t *exception
  SIDL_F90_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F90_STR(name),
      SIDL_F90_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self,
      _proxy_name,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = ((_proxy_retval == TRUE) ? SIDL_F90_TRUE : SIDL_F90_FALSE);
  }
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran90Symbol(bhypre_structvector_getclassinfo_m,BHYPRE_STRUCTVECTOR_GETCLASSINFO_M,bHYPRE_StructVector_getClassInfo_m)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *retval = (ptrdiff_t)_proxy_retval;
  }
}

/*
 *  Set the grid on which vectors are defined. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_setgrid_m,BHYPRE_STRUCTVECTOR_SETGRID_M,bHYPRE_StructVector_SetGrid_m)
(
  int64_t *self,
  int64_t *grid,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct bHYPRE_StructGrid__object* _proxy_grid = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_grid =
    (struct bHYPRE_StructGrid__object*)
    (ptrdiff_t)(*grid);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetGrid))(
      _proxy_self,
      _proxy_grid,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 *  Set the number of ghost zones, separately on the lower and upper sides
 * for each dimension.
 * "num_ghost" is an array of size "dim2", twice the number of dimensions. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_setnumghost_m,BHYPRE_STRUCTVECTOR_SETNUMGHOST_M,bHYPRE_StructVector_SetNumGhost_m)
(
  int64_t *self,
  int32_t *num_ghost,
  int32_t *dim2,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_int__array _alt_num_ghost;
  struct sidl_int__array* _proxy_num_ghost = &_alt_num_ghost;
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1];
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  num_ghost_upper[0] = (*dim2)-1;
  sidl_int__array_init(num_ghost, _proxy_num_ghost, 1, num_ghost_lower,
    num_ghost_upper, num_ghost_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetNumGhost))(
      _proxy_self,
      _proxy_num_ghost,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}
/*
 *  Set the number of ghost zones, separately on the lower and upper sides
 * for each dimension.
 * "num_ghost" is an array of size "dim2", twice the number of dimensions. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_setnumghost_a,BHYPRE_STRUCTVECTOR_SETNUMGHOST_A,bHYPRE_StructVector_SetNumGhost_a)
(
  int64_t *self,
  struct sidl_fortran_array *num_ghost,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_num_ghost = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_num_ghost =
    (struct sidl_int__array*)
    (ptrdiff_t)(num_ghost->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetNumGhost))(
      _proxy_self,
      _proxy_num_ghost,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 *  Set the value of a single vector coefficient, given by "grid_index".
 * "grid_index" is an array of size "dim", where dim is the number
 * of dimensions. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_setvalue_m,BHYPRE_STRUCTVECTOR_SETVALUE_M,bHYPRE_StructVector_SetValue_m)
(
  int64_t *self,
  int32_t *grid_index,
  int32_t *dim,
  double *value,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_int__array _alt_grid_index;
  struct sidl_int__array* _proxy_grid_index = &_alt_grid_index;
  int32_t grid_index_lower[1], grid_index_upper[1], grid_index_stride[1];
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  grid_index_upper[0] = (*dim)-1;
  sidl_int__array_init(grid_index, _proxy_grid_index, 1, grid_index_lower,
    grid_index_upper, grid_index_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetValue))(
      _proxy_self,
      _proxy_grid_index,
      *value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}
/*
 *  Set the value of a single vector coefficient, given by "grid_index".
 * "grid_index" is an array of size "dim", where dim is the number
 * of dimensions. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_setvalue_a,BHYPRE_STRUCTVECTOR_SETVALUE_A,bHYPRE_StructVector_SetValue_a)
(
  int64_t *self,
  struct sidl_fortran_array *grid_index,
  double *value,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_grid_index = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_grid_index =
    (struct sidl_int__array*)
    (ptrdiff_t)(grid_index->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetValue))(
      _proxy_self,
      _proxy_grid_index,
      *value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 *  Set the values of all vector coefficient for grid points in a box.
 * The box is defined by its lower and upper corners in the grid.
 * "ilower" and "iupper" are arrays of size "dim", where dim is the
 * number of dimensions.  The "values" array has size "nvalues", which
 * is the number of grid points in the box. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_setboxvalues_m,BHYPRE_STRUCTVECTOR_SETBOXVALUES_M,bHYPRE_StructVector_SetBoxValues_m)
(
  int64_t *self,
  int32_t *ilower,
  int32_t *iupper,
  int32_t *dim,
  double *values,
  int32_t *nvalues,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_int__array _alt_ilower;
  struct sidl_int__array* _proxy_ilower = &_alt_ilower;
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1];
  struct sidl_int__array _alt_iupper;
  struct sidl_int__array* _proxy_iupper = &_alt_iupper;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1];
  struct sidl_double__array _alt_values;
  struct sidl_double__array* _proxy_values = &_alt_values;
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  ilower_upper[0] = (*dim)-1;
  sidl_int__array_init(ilower, _proxy_ilower, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = (*dim)-1;
  sidl_int__array_init(iupper, _proxy_iupper, 1, iupper_lower, iupper_upper,
    iupper_stride);
  values_upper[0] = (*nvalues)-1;
  sidl_double__array_init(values, _proxy_values, 1, values_lower, values_upper,
    values_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetBoxValues))(
      _proxy_self,
      _proxy_ilower,
      _proxy_iupper,
      _proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}
/*
 *  Set the values of all vector coefficient for grid points in a box.
 * The box is defined by its lower and upper corners in the grid.
 * "ilower" and "iupper" are arrays of size "dim", where dim is the
 * number of dimensions.  The "values" array has size "nvalues", which
 * is the number of grid points in the box. 
 */

void
SIDLFortran90Symbol(bhypre_structvector_setboxvalues_a,BHYPRE_STRUCTVECTOR_SETBOXVALUES_A,bHYPRE_StructVector_SetBoxValues_a)
(
  int64_t *self,
  struct sidl_fortran_array *ilower,
  struct sidl_fortran_array *iupper,
  struct sidl_fortran_array *values,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_int__array* _proxy_ilower = NULL;
  struct sidl_int__array* _proxy_iupper = NULL;
  struct sidl_double__array* _proxy_values = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_ilower =
    (struct sidl_int__array*)
    (ptrdiff_t)(ilower->d_ior);
  _proxy_iupper =
    (struct sidl_int__array*)
    (ptrdiff_t)(iupper->d_ior);
  _proxy_values =
    (struct sidl_double__array*)
    (ptrdiff_t)(values->d_ior);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetBoxValues))(
      _proxy_self,
      _proxy_ilower,
      _proxy_iupper,
      _proxy_values,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

void
SIDLFortran90Symbol(bhypre_structvector_setcommunicator_m,BHYPRE_STRUCTVECTOR_SETCOMMUNICATOR_M,bHYPRE_StructVector_SetCommunicator_m)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self,
      _proxy_mpi_comm,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

void
SIDLFortran90Symbol(bhypre_structvector_destroy_m,BHYPRE_STRUCTVECTOR_DESTROY_M,bHYPRE_StructVector_Destroy_m)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_Destroy))(
    _proxy_self,
    &_proxy_exception
  );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */

void
SIDLFortran90Symbol(bhypre_structvector_initialize_m,BHYPRE_STRUCTVECTOR_INITIALIZE_M,bHYPRE_StructVector_Initialize_m)
(
  int64_t *self,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Initialize))(
      _proxy_self,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */

void
SIDLFortran90Symbol(bhypre_structvector_assemble_m,BHYPRE_STRUCTVECTOR_ASSEMBLE_M,bHYPRE_StructVector_Assemble_m)
(
  int64_t *self,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Assemble))(
      _proxy_self,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Set {\tt self} to 0.
 */

void
SIDLFortran90Symbol(bhypre_structvector_clear_m,BHYPRE_STRUCTVECTOR_CLEAR_M,bHYPRE_StructVector_Clear_m)
(
  int64_t *self,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Clear))(
      _proxy_self,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Copy data from x into {\tt self}.
 */

void
SIDLFortran90Symbol(bhypre_structvector_copy_m,BHYPRE_STRUCTVECTOR_COPY_M,bHYPRE_StructVector_Copy_m)
(
  int64_t *self,
  int64_t *x,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Copy))(
      _proxy_self,
      _proxy_x,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Create an {\tt x} compatible with {\tt self}.
 * The new vector's data is not specified.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 */

void
SIDLFortran90Symbol(bhypre_structvector_clone_m,BHYPRE_STRUCTVECTOR_CLONE_M,bHYPRE_StructVector_Clone_m)
(
  int64_t *self,
  int64_t *x,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Clone))(
      _proxy_self,
      &_proxy_x,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
    *x = (ptrdiff_t)_proxy_x;
  }
}

/*
 * Scale {\tt self} by {\tt a}.
 */

void
SIDLFortran90Symbol(bhypre_structvector_scale_m,BHYPRE_STRUCTVECTOR_SCALE_M,bHYPRE_StructVector_Scale_m)
(
  int64_t *self,
  double *a,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Scale))(
      _proxy_self,
      *a,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 */

void
SIDLFortran90Symbol(bhypre_structvector_dot_m,BHYPRE_STRUCTVECTOR_DOT_M,bHYPRE_StructVector_Dot_m)
(
  int64_t *self,
  int64_t *x,
  double *d,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Dot))(
      _proxy_self,
      _proxy_x,
      d,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

/*
 * Add {\tt a}{\tt x} to {\tt self}.
 */

void
SIDLFortran90Symbol(bhypre_structvector_axpy_m,BHYPRE_STRUCTVECTOR_AXPY_M,bHYPRE_StructVector_Axpy_m)
(
  int64_t *self,
  double *a,
  int64_t *x,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructVector__epv *_epv = NULL;
  struct bHYPRE_StructVector__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructVector__object*)
    (ptrdiff_t)(*self);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Axpy))(
      _proxy_self,
      *a,
      _proxy_x,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
}

void
SIDLFortran90Symbol(bhypre_structvector__array_createcol_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_CREATECOL_M,
                  bHYPRE_StructVector__array_createCol_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_createrow_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_CREATEROW_M,
                  bHYPRE_StructVector__array_createRow_m)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_create1d_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_CREATE1D_M,
                  bHYPRE_StructVector__array_create1d_m)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_create2dcol_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_CREATE2DCOL_M,
                  bHYPRE_StructVector__array_create2dCol_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_create2drow_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_CREATE2DROW_M,
                  bHYPRE_StructVector__array_create2dRow_m)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_addref_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_ADDREF_M,
                  bHYPRE_StructVector__array_addRef_m)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_deleteref_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_DELETEREF_M,
                  bHYPRE_StructVector__array_deleteRef_m)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get1_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET1_M,
                  bHYPRE_StructVector__array_get1_m)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get1((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get2_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET2_M,
                  bHYPRE_StructVector__array_get2_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get2((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get3_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET3_M,
                  bHYPRE_StructVector__array_get3_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get3((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get4_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET4_M,
                  bHYPRE_StructVector__array_get4_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get4((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get5_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET5_M,
                  bHYPRE_StructVector__array_get5_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get5((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get6_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET6_M,
                  bHYPRE_StructVector__array_get6_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get6((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get7_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET7_M,
                  bHYPRE_StructVector__array_get7_m)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int32_t *i5, 
   int32_t *i6, 
   int32_t *i7, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get7((const struct sidl_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4, *i5, *i6, *i7);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_get_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_GET_M,
                  bHYPRE_StructVector__array_get_m)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set1_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET1_M,
                  bHYPRE_StructVector__array_set1_m)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set2_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET2_M,
                  bHYPRE_StructVector__array_set2_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set3_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET3_M,
                  bHYPRE_StructVector__array_set3_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set4_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET4_M,
                  bHYPRE_StructVector__array_set4_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set5_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET5_M,
                  bHYPRE_StructVector__array_set5_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int64_t *value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set6_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET6_M,
                  bHYPRE_StructVector__array_set6_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int64_t *value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set7_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET7_M,
                  bHYPRE_StructVector__array_set7_m)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int32_t *i5,
   int32_t *i6,
   int32_t *i7,
   int64_t *value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, *i5, *i6, *i7,
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_set_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SET_M,
                  bHYPRE_StructVector__array_set_m)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_dimen_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_DIMEN_M,
                  bHYPRE_StructVector__array_dimen_m)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_lower_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_LOWER_M,
                  bHYPRE_StructVector__array_lower_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_upper_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_UPPER_M,
                  bHYPRE_StructVector__array_upper_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_length_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_LENGTH_M,
                  bHYPRE_StructVector__array_length_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_stride_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_STRIDE_M,
                  bHYPRE_StructVector__array_stride_m)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_iscolumnorder_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_ISCOLUMNORDER_M,
                  bHYPRE_StructVector__array_isColumnOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_isroworder_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_ISROWORDER_M,
                  bHYPRE_StructVector__array_isRowOrder_m)
  (int64_t *array,
   SIDL_F90_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_copy_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_COPY_M,
                  bHYPRE_StructVector__array_copy_m)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_smartcopy_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SMARTCOPY_M,
                  bHYPRE_StructVector__array_smartCopy_m)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_slice_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_SLICE_M,
                  bHYPRE_StructVector__array_slice_m)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_slice((struct sidl_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran90Symbol(bhypre_structvector__array_ensure_m,
                  BHYPRE_STRUCTVECTOR__ARRAY_ENSURE_M,
                  bHYPRE_StructVector__array_ensure_m)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_ensure((struct sidl_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#include "sidl_Exception.h"

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

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 10;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct bHYPRE_StructVector__epv s_rem_epv__bhypre_structvector;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct bHYPRE_StructVectorView__epv s_rem_epv__bhypre_structvectorview;

static struct bHYPRE_Vector__epv s_rem_epv__bhypre_vector;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_StructVector__cast(
  struct bHYPRE_StructVector__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "bHYPRE.StructVectorView");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_bhypre_structvectorview);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.ProblemDefinition");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_problemdefinition);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.MatrixVectorView");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_matrixvectorview);
        return cast;
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "bHYPRE.StructVector");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseClass");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.Vector");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_vector);
        return cast;
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.BaseInterface");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
        return cast;
      }
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct bHYPRE_StructVector__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_StructVector__delete(
  struct bHYPRE_StructVector__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_StructVector__getURL(
  struct bHYPRE_StructVector__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE_StructVector__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_StructVector__raddRef(
  struct bHYPRE_StructVector__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_StructVector__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    sidl_BaseInterface throwaway_exception = NULL;
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
      &throwaway_exception);
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_bHYPRE_StructVector__isRemote(
    struct bHYPRE_StructVector__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_StructVector__set_hooks(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ sidl_bool on,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector._set_hooks.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE_StructVector__exec(
  struct bHYPRE_StructVector__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_StructVector_addRef(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_StructVector__remote* r_obj = (struct 
      bHYPRE_StructVector__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_StructVector_deleteRef(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_StructVector__remote* r_obj = (struct 
      bHYPRE_StructVector__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_bHYPRE_StructVector_isSame(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.isSame.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE_StructVector_isType(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.isType.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_bHYPRE_StructVector_getClassInfo(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.getClassInfo.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetGrid */
static int32_t
remote_bHYPRE_StructVector_SetGrid(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ struct bHYPRE_StructGrid__object* grid,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetGrid", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(grid){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)grid,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "grid", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "grid", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.SetGrid.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetNumGhost */
static int32_t
remote_bHYPRE_StructVector_SetNumGhost(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetNumGhost", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "num_ghost", num_ghost,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.SetNumGhost.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetValue */
static int32_t
remote_bHYPRE_StructVector_SetValue(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in rarray[dim] */ struct sidl_int__array* grid_index,
  /* in */ double value,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetValue", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "grid_index", grid_index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.SetValue.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetBoxValues */
static int32_t
remote_bHYPRE_StructVector_SetBoxValues(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.SetBoxValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE_StructVector_SetCommunicator(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetCommunicator", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(mpi_comm){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)mpi_comm,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "mpi_comm", _url,
        _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "mpi_comm", NULL,
        _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.SetCommunicator.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Destroy */
static void
remote_bHYPRE_StructVector_Destroy(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Destroy.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:Initialize */
static int32_t
remote_bHYPRE_StructVector_Initialize(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Initialize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Initialize.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Assemble */
static int32_t
remote_bHYPRE_StructVector_Assemble(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Assemble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Assemble.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Clear */
static int32_t
remote_bHYPRE_StructVector_Clear(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Clear", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Clear.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Copy */
static int32_t
remote_bHYPRE_StructVector_Copy(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ struct bHYPRE_Vector__object* x,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Copy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Copy.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Clone */
static int32_t
remote_bHYPRE_StructVector_Clone(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* out */ struct bHYPRE_Vector__object** x,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* x_str= NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Clone", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Clone.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "x", &x_str, _ex);SIDL_CHECK(*_ex);
    *x = bHYPRE_Vector__connectI(x_str, FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Scale */
static int32_t
remote_bHYPRE_StructVector_Scale(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ double a,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Scale", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "a", a, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Scale.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Dot */
static int32_t
remote_bHYPRE_StructVector_Dot(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ struct bHYPRE_Vector__object* x,
  /* out */ double* d,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Dot", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Dot.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "d", d, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Axpy */
static int32_t
remote_bHYPRE_StructVector_Axpy(
  /* in */ struct bHYPRE_StructVector__object* self ,
  /* in */ double a,
  /* in */ struct bHYPRE_Vector__object* x,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_StructVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Axpy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "a", a, _ex);SIDL_CHECK(*_ex);
    if(x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructVector.Axpy.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_StructVector__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_StructVector__epv*      epv = &s_rem_epv__bhypre_structvector;
  struct bHYPRE_MatrixVectorView__epv*  e0  = 
    &s_rem_epv__bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__epv* e1  = 
    &s_rem_epv__bhypre_problemdefinition;
  struct bHYPRE_StructVectorView__epv*  e2  = 
    &s_rem_epv__bhypre_structvectorview;
  struct bHYPRE_Vector__epv*            e3  = &s_rem_epv__bhypre_vector;
  struct sidl_BaseClass__epv*           e4  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e5  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast                = remote_bHYPRE_StructVector__cast;
  epv->f__delete              = remote_bHYPRE_StructVector__delete;
  epv->f__exec                = remote_bHYPRE_StructVector__exec;
  epv->f__getURL              = remote_bHYPRE_StructVector__getURL;
  epv->f__raddRef             = remote_bHYPRE_StructVector__raddRef;
  epv->f__isRemote            = remote_bHYPRE_StructVector__isRemote;
  epv->f__set_hooks           = remote_bHYPRE_StructVector__set_hooks;
  epv->f__ctor                = NULL;
  epv->f__ctor2               = NULL;
  epv->f__dtor                = NULL;
  epv->f_addRef               = remote_bHYPRE_StructVector_addRef;
  epv->f_deleteRef            = remote_bHYPRE_StructVector_deleteRef;
  epv->f_isSame               = remote_bHYPRE_StructVector_isSame;
  epv->f_isType               = remote_bHYPRE_StructVector_isType;
  epv->f_getClassInfo         = remote_bHYPRE_StructVector_getClassInfo;
  epv->f_SetGrid              = remote_bHYPRE_StructVector_SetGrid;
  epv->f_SetNumGhost          = remote_bHYPRE_StructVector_SetNumGhost;
  epv->f_SetValue             = remote_bHYPRE_StructVector_SetValue;
  epv->f_SetBoxValues         = remote_bHYPRE_StructVector_SetBoxValues;
  epv->f_SetCommunicator      = remote_bHYPRE_StructVector_SetCommunicator;
  epv->f_Destroy              = remote_bHYPRE_StructVector_Destroy;
  epv->f_Initialize           = remote_bHYPRE_StructVector_Initialize;
  epv->f_Assemble             = remote_bHYPRE_StructVector_Assemble;
  epv->f_Clear                = remote_bHYPRE_StructVector_Clear;
  epv->f_Copy                 = remote_bHYPRE_StructVector_Copy;
  epv->f_Clone                = remote_bHYPRE_StructVector_Clone;
  epv->f_Scale                = remote_bHYPRE_StructVector_Scale;
  epv->f_Dot                  = remote_bHYPRE_StructVector_Dot;
  epv->f_Axpy                 = remote_bHYPRE_StructVector_Axpy;

  e0->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e0->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e0->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e0->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e1->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e1->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e1->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e2->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_SetGrid         = (int32_t (*)(void*,struct bHYPRE_StructGrid__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetGrid;
  e2->f_SetNumGhost     = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetNumGhost;
  e2->f_SetValue        = (int32_t (*)(void*,struct sidl_int__array*,double,
    struct sidl_BaseInterface__object **)) epv->f_SetValue;
  e2->f_SetBoxValues    = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetBoxValues;
  e2->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e2->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e2->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e2->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e3->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e3->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e3->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e3->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e3->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_Clear        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Clear;
  e3->f_Copy         = (int32_t (*)(void*,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Copy;
  e3->f_Clone        = (int32_t (*)(void*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_Clone;
  e3->f_Scale        = (int32_t (*)(void*,double,
    struct sidl_BaseInterface__object **)) epv->f_Scale;
  e3->f_Dot          = (int32_t (*)(void*,struct bHYPRE_Vector__object*,double*,
    struct sidl_BaseInterface__object **)) epv->f_Dot;
  e3->f_Axpy         = (int32_t (*)(void*,double,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Axpy;
  e3->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e4->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e4->f__delete      = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e4->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e4->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e4->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e4->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e4->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e4->f_addRef       = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e4->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e5->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e5->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e5->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e5->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e5->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e5->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e5->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e5->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e5->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e5->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e5->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_StructVector__object*
bHYPRE_StructVector__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_StructVector__object* self;

  struct bHYPRE_StructVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_StructVector__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = NULL;
  *_ex = NULL;
  if(url == NULL) {return NULL;}
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex); SIDL_CHECK(*_ex);
    return bHYPRE_StructVector__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_StructVector__object*) malloc(
      sizeof(struct bHYPRE_StructVector__object));

  r_obj =
    (struct bHYPRE_StructVector__remote*) malloc(
      sizeof(struct bHYPRE_StructVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                               self;
  s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_structvectorview.d_epv    = &s_rem_epv__bhypre_structvectorview;
  s0->d_bhypre_structvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_structvector;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct bHYPRE_StructVector__object*
bHYPRE_StructVector__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_StructVector__object* self;

  struct bHYPRE_StructVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_StructVector__remote* r_obj;
  self =
    (struct bHYPRE_StructVector__object*) malloc(
      sizeof(struct bHYPRE_StructVector__object));

  r_obj =
    (struct bHYPRE_StructVector__remote*) malloc(
      sizeof(struct bHYPRE_StructVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                               self;
  s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_structvectorview.d_epv    = &s_rem_epv__bhypre_structvectorview;
  s0->d_bhypre_structvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_structvector;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_StructVector__object*
bHYPRE_StructVector__remoteCreate(const char *url, sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct bHYPRE_StructVector__object* self;

  struct bHYPRE_StructVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_StructVector__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.StructVector",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_StructVector__object*) malloc(
      sizeof(struct bHYPRE_StructVector__object));

  r_obj =
    (struct bHYPRE_StructVector__remote*) malloc(
      sizeof(struct bHYPRE_StructVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                               self;
  s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_structvectorview.d_epv    = &s_rem_epv__bhypre_structvectorview;
  s0->d_bhypre_structvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_structvector;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance,
    &_throwaway_exception); }
  return NULL;
}
/*
 * Cast method for interface and class type conversions.
 */

struct bHYPRE_StructVector__object*
bHYPRE_StructVector__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_StructVector__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.StructVector",
      (void*)bHYPRE_StructVector__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_StructVector__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.StructVector", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_StructVector__object*
bHYPRE_StructVector__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return bHYPRE_StructVector__remoteConnect(url, ar, _ex);
}

