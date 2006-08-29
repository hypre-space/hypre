/*
 * File:          bHYPRE_StructDiagScale_fStub.c
 * Symbol:        bHYPRE.StructDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.StructDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

/*
 * Symbol "bHYPRE.StructDiagScale" (version 1.0.0)
 * 
 * Diagonal scaling preconditioner for STruct matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 */

#ifndef included_bHYPRE_StructDiagScale_fStub_h
#include "bHYPRE_StructDiagScale_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
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
#include "bHYPRE_StructDiagScale_IOR.h"
#include "bHYPRE_MPICommunicator_IOR.h"
#include "bHYPRE_Operator_IOR.h"
#include "bHYPRE_StructMatrix_IOR.h"
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
#ifndef included_bHYPRE_Operator_fStub_h
#include "bHYPRE_Operator_fStub.h"
#endif
#ifndef included_bHYPRE_Solver_fStub_h
#include "bHYPRE_Solver_fStub.h"
#endif
#ifndef included_bHYPRE_StructDiagScale_fStub_h
#include "bHYPRE_StructDiagScale_fStub.h"
#endif
#ifndef included_bHYPRE_StructMatrix_fStub_h
#include "bHYPRE_StructMatrix_fStub.h"
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

static struct bHYPRE_StructDiagScale__object* 
  bHYPRE_StructDiagScale__remoteCreate(const char* url,
  sidl_BaseInterface *_ex);
static struct bHYPRE_StructDiagScale__object* 
  bHYPRE_StructDiagScale__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct bHYPRE_StructDiagScale__object* 
  bHYPRE_StructDiagScale__IHConnect(struct sidl_rmi_InstanceHandle__object 
  *instance, struct sidl_BaseInterface__object **_ex);
/*
 * Return pointer to internal IOR functions.
 */

static const struct bHYPRE_StructDiagScale__external* _getIOR(void)
{
  static const struct bHYPRE_StructDiagScale__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = bHYPRE_StructDiagScale__externals();
#else
    _ior = (struct 
      bHYPRE_StructDiagScale__external*)sidl_dynamicLoadIOR(
      "bHYPRE.StructDiagScale","bHYPRE_StructDiagScale__externals") ;
    sidl_checkIORVersion("bHYPRE.StructDiagScale", _ior->d_ior_major_version,
      _ior->d_ior_minor_version, 0, 10);
#endif
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct bHYPRE_StructDiagScale__sepv* _getSEPV(void)
{
  static const struct bHYPRE_StructDiagScale__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale__create_f,BHYPRE_STRUCTDIAGSCALE__CREATE_F,bHYPRE_StructDiagScale__create_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__createremote_f,BHYPRE_STRUCTDIAGSCALE__CREATEREMOTE_F,bHYPRE_StructDiagScale__createRemote_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = bHYPRE_StructDiagScale__remoteCreate(_proxy_url,
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
SIDLFortran77Symbol(bhypre_structdiagscale__wrapobj_f,BHYPRE_STRUCTDIAGSCALE__WRAPOBJ_F,bHYPRE_StructDiagScale__wrapObj_f)
(
  int64_t * private_data,
  int64_t *self,
  int64_t *exception
)
{
  struct sidl_BaseInterface__object *_ior_exception = NULL;
  *self = (ptrdiff_t) 
    (*(_getIOR()->createObject))((void*)(ptrdiff_t)*private_data,
    &_ior_exception);
  *exception = (ptrdiff_t)_ior_exception;
  if (_ior_exception) *self = 0;
}

/*
 * Remote Connector for the class.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale__connect_f,BHYPRE_STRUCTDIAGSCALE__CONNECT_F,bHYPRE_StructDiagScale__connect_f)
(
  int64_t *self,
  SIDL_F77_String url
  SIDL_F77_STR_NEAR_LEN_DECL(url),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(url)
)
{
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_url = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_url =
    sidl_copy_fortran_str(SIDL_F77_STR(url),
      SIDL_F77_STR_LEN(url));
  _proxy_self = bHYPRE_StructDiagScale__remoteConnect(_proxy_url, 1,
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
SIDLFortran77Symbol(bhypre_structdiagscale__cast_f,BHYPRE_STRUCTDIAGSCALE__CAST_F,bHYPRE_StructDiagScale__cast_f)
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
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.StructDiagScale",
      (void*)bHYPRE_StructDiagScale__IHConnect, &proxy_exception);
    SIDL_CHECK(proxy_exception);
    connect_loaded = 1;
  }

  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.StructDiagScale", &proxy_exception);
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
SIDLFortran77Symbol(bhypre_structdiagscale__cast2_f,BHYPRE_STRUCTDIAGSCALE__CAST2_F,bHYPRE_StructDiagScale__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
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
SIDLFortran77Symbol(bhypre_structdiagscale__exec_f,BHYPRE_STRUCTDIAGSCALE__EXEC_F,bHYPRE_StructDiagScale__exec_f)
(
  int64_t *self,
  SIDL_F77_String methodName
  SIDL_F77_STR_NEAR_LEN_DECL(methodName),
  int64_t *inArgs,
  int64_t *outArgs,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(methodName)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_methodName = NULL;
  struct sidl_rmi_Call__object* _proxy_inArgs = NULL;
  struct sidl_rmi_Return__object* _proxy_outArgs = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_methodName =
    sidl_copy_fortran_str(SIDL_F77_STR(methodName),
      SIDL_F77_STR_LEN(methodName));
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
SIDLFortran77Symbol(bhypre_structdiagscale__geturl_f,BHYPRE_STRUCTDIAGSCALE__GETURL_F,bHYPRE_StructDiagScale__getURL_f)
(
  int64_t *self,
  SIDL_F77_String retval
  SIDL_F77_STR_NEAR_LEN_DECL(retval),
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(retval)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
      SIDL_F77_STR(retval),
      SIDL_F77_STR_LEN(retval),
      _proxy_retval);
  }
  free((void *)_proxy_retval);
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale__isremote_f,BHYPRE_STRUCTDIAGSCALE__ISREMOTE_F,bHYPRE_StructDiagScale__isRemote_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
}


/*
 * TRUE if this object is remote, false if local
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale__islocal_f,BHYPRE_STRUCTDIAGSCALE__ISLOCAL_F,bHYPRE_StructDiagScale__isLocal_f)
(
  int64_t *self,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
}


/*
 * Method to set whether or not method hooks should be invoked.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale__set_hooks_f,BHYPRE_STRUCTDIAGSCALE__SET_HOOKS_F,bHYPRE_StructDiagScale__set_hooks_f)
(
  int64_t *self,
  SIDL_F77_Bool *on,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_on = ((*on == SIDL_F77_TRUE) ? TRUE : FALSE);
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
SIDLFortran77Symbol(bhypre_structdiagscale__set_hooks_static_f,BHYPRE_STRUCTDIAGSCALE__SET_HOOKS_STATIC_F,bHYPRE_StructDiagScale__set_hooks_static_f)
(
  SIDL_F77_Bool *on,
  int64_t *exception
)
{
  const struct bHYPRE_StructDiagScale__sepv *_epv = _getSEPV();
  sidl_bool _proxy_on;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_on = ((*on == SIDL_F77_TRUE) ? TRUE : FALSE);
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
 * Method:  Create[]
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_create_f,BHYPRE_STRUCTDIAGSCALE_CREATE_F,bHYPRE_StructDiagScale_Create_f)
(
  int64_t *mpi_comm,
  int64_t *A,
  int64_t *retval,
  int64_t *exception
)
{
  const struct bHYPRE_StructDiagScale__sepv *_epv = _getSEPV();
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  struct bHYPRE_StructMatrix__object* _proxy_A = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _proxy_A =
    (struct bHYPRE_StructMatrix__object*)
    (ptrdiff_t)(*A);
  _proxy_retval = 
    (*(_epv->f_Create))(
      _proxy_mpi_comm,
      _proxy_A,
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
SIDLFortran77Symbol(bhypre_structdiagscale_addref_f,BHYPRE_STRUCTDIAGSCALE_ADDREF_F,bHYPRE_StructDiagScale_addRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
SIDLFortran77Symbol(bhypre_structdiagscale_deleteref_f,BHYPRE_STRUCTDIAGSCALE_DELETEREF_F,bHYPRE_StructDiagScale_deleteRef_f)
(
  int64_t *self,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
SIDLFortran77Symbol(bhypre_structdiagscale_issame_f,BHYPRE_STRUCTDIAGSCALE_ISSAME_F,bHYPRE_StructDiagScale_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_istype_f,BHYPRE_STRUCTDIAGSCALE_ISTYPE_F,bHYPRE_StructDiagScale_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
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
    *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  }
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_getclassinfo_f,BHYPRE_STRUCTDIAGSCALE_GETCLASSINFO_F,bHYPRE_StructDiagScale_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setoperator_f,BHYPRE_STRUCTDIAGSCALE_SETOPERATOR_F,bHYPRE_StructDiagScale_SetOperator_f)
(
  int64_t *self,
  int64_t *A,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Operator__object* _proxy_A = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_A =
    (struct bHYPRE_Operator__object*)
    (ptrdiff_t)(*A);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetOperator))(
      _proxy_self,
      _proxy_A,
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
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_settolerance_f,BHYPRE_STRUCTDIAGSCALE_SETTOLERANCE_F,bHYPRE_StructDiagScale_SetTolerance_f)
(
  int64_t *self,
  double *tolerance,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetTolerance))(
      _proxy_self,
      *tolerance,
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
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setmaxiterations_f,BHYPRE_STRUCTDIAGSCALE_SETMAXITERATIONS_F,bHYPRE_StructDiagScale_SetMaxIterations_f)
(
  int64_t *self,
  int32_t *max_iterations,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetMaxIterations))(
      _proxy_self,
      *max_iterations,
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
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setlogging_f,BHYPRE_STRUCTDIAGSCALE_SETLOGGING_F,bHYPRE_StructDiagScale_SetLogging_f)
(
  int64_t *self,
  int32_t *level,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLogging))(
      _proxy_self,
      *level,
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
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setprintlevel_f,BHYPRE_STRUCTDIAGSCALE_SETPRINTLEVEL_F,bHYPRE_StructDiagScale_SetPrintLevel_f)
(
  int64_t *self,
  int32_t *level,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetPrintLevel))(
      _proxy_self,
      *level,
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
 * (Optional) Return the number of iterations taken.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_getnumiterations_f,BHYPRE_STRUCTDIAGSCALE_GETNUMITERATIONS_F,bHYPRE_StructDiagScale_GetNumIterations_f)
(
  int64_t *self,
  int32_t *num_iterations,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetNumIterations))(
      _proxy_self,
      num_iterations,
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
 * (Optional) Return the norm of the relative residual.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_getrelresidualnorm_f,BHYPRE_STRUCTDIAGSCALE_GETRELRESIDUALNORM_F,bHYPRE_StructDiagScale_GetRelResidualNorm_f)
(
  int64_t *self,
  double *norm,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRelResidualNorm))(
      _proxy_self,
      norm,
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
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setcommunicator_f,BHYPRE_STRUCTDIAGSCALE_SETCOMMUNICATOR_F,bHYPRE_StructDiagScale_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
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
 * Set the int parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setintparameter_f,BHYPRE_STRUCTDIAGSCALE_SETINTPARAMETER_F,bHYPRE_StructDiagScale_SetIntParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntParameter))(
      _proxy_self,
      _proxy_name,
      *value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * Set the double parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setdoubleparameter_f,BHYPRE_STRUCTDIAGSCALE_SETDOUBLEPARAMETER_F,bHYPRE_StructDiagScale_SetDoubleParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleParameter))(
      _proxy_self,
      _proxy_name,
      *value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * Set the string parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setstringparameter_f,BHYPRE_STRUCTDIAGSCALE_SETSTRINGPARAMETER_F,bHYPRE_StructDiagScale_SetStringParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_String value
  SIDL_F77_STR_NEAR_LEN_DECL(value),
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
  SIDL_F77_STR_FAR_LEN_DECL(value)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  char* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    sidl_copy_fortran_str(SIDL_F77_STR(value),
      SIDL_F77_STR_LEN(value));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetStringParameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
  free((void *)_proxy_value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setintarray1parameter_f,BHYPRE_STRUCTDIAGSCALE_SETINTARRAY1PARAMETER_F,bHYPRE_StructDiagScale_SetIntArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *nvalues,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array _alt_value;
  struct sidl_int__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  value_upper[0] = (*nvalues)-1;
  sidl_int__array_init(value, _proxy_value, 1, value_lower, value_upper,
    value_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setintarray2parameter_f,BHYPRE_STRUCTDIAGSCALE_SETINTARRAY2PARAMETER_F,bHYPRE_StructDiagScale_SetIntArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct sidl_int__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setdoublearray1parameter_f,BHYPRE_STRUCTDIAGSCALE_SETDOUBLEARRAY1PARAMETER_F,bHYPRE_StructDiagScale_SetDoubleArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *nvalues,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array _alt_value;
  struct sidl_double__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  value_upper[0] = (*nvalues)-1;
  sidl_double__array_init(value, _proxy_value, 1, value_lower, value_upper,
    value_stride);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray1Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setdoublearray2parameter_f,BHYPRE_STRUCTDIAGSCALE_SETDOUBLEARRAY2PARAMETER_F,bHYPRE_StructDiagScale_SetDoubleArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array* _proxy_value = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _proxy_value =
    (struct sidl_double__array*)
    (ptrdiff_t)(*value);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleArray2Parameter))(
      _proxy_self,
      _proxy_name,
      _proxy_value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * Set the int parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_getintvalue_f,BHYPRE_STRUCTDIAGSCALE_GETINTVALUE_F,bHYPRE_StructDiagScale_GetIntValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetIntValue))(
      _proxy_self,
      _proxy_name,
      value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * Get the double parameter associated with {\tt name}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_getdoublevalue_f,BHYPRE_STRUCTDIAGSCALE_GETDOUBLEVALUE_F,bHYPRE_StructDiagScale_GetDoubleValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval,
  int64_t *exception
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetDoubleValue))(
      _proxy_self,
      _proxy_name,
      value,
      &_proxy_exception
    );
  if (_proxy_exception) {
    *exception = (ptrdiff_t)_proxy_exception;
  }
  else {
    *exception = (ptrdiff_t)NULL;
  }
  free((void *)_proxy_name);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_setup_f,BHYPRE_STRUCTDIAGSCALE_SETUP_F,bHYPRE_StructDiagScale_Setup_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Setup))(
      _proxy_self,
      _proxy_b,
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
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_apply_f,BHYPRE_STRUCTDIAGSCALE_APPLY_F,bHYPRE_StructDiagScale_Apply_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Apply))(
      _proxy_self,
      _proxy_b,
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
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

void
SIDLFortran77Symbol(bhypre_structdiagscale_applyadjoint_f,BHYPRE_STRUCTDIAGSCALE_APPLYADJOINT_F,bHYPRE_StructDiagScale_ApplyAdjoint_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval,
  int64_t *exception
)
{
  struct bHYPRE_StructDiagScale__epv *_epv = NULL;
  struct bHYPRE_StructDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  struct sidl_BaseInterface__object* _proxy_exception = NULL;
  _proxy_self =
    (struct bHYPRE_StructDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_b =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*b);
  _proxy_x =
    (struct bHYPRE_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_ApplyAdjoint))(
      _proxy_self,
      _proxy_b,
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

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_createcol_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_CREATECOL_F,
                  bHYPRE_StructDiagScale__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_createrow_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_CREATEROW_F,
                  bHYPRE_StructDiagScale__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_create1d_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_CREATE1D_F,
                  bHYPRE_StructDiagScale__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_create2dcol_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_CREATE2DCOL_F,
                  bHYPRE_StructDiagScale__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_create2drow_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_CREATE2DROW_F,
                  bHYPRE_StructDiagScale__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_addref_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_ADDREF_F,
                  bHYPRE_StructDiagScale__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_deleteref_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_DELETEREF_F,
                  bHYPRE_StructDiagScale__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_get1_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET1_F,
                  bHYPRE_StructDiagScale__array_get1_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_get2_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET2_F,
                  bHYPRE_StructDiagScale__array_get2_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_get3_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET3_F,
                  bHYPRE_StructDiagScale__array_get3_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_get4_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET4_F,
                  bHYPRE_StructDiagScale__array_get4_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_get5_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET5_F,
                  bHYPRE_StructDiagScale__array_get5_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_get6_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET6_F,
                  bHYPRE_StructDiagScale__array_get6_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_get7_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET7_F,
                  bHYPRE_StructDiagScale__array_get7_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_get_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_GET_F,
                  bHYPRE_StructDiagScale__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_set1_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET1_F,
                  bHYPRE_StructDiagScale__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_set2_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET2_F,
                  bHYPRE_StructDiagScale__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_set3_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET3_F,
                  bHYPRE_StructDiagScale__array_set3_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_set4_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET4_F,
                  bHYPRE_StructDiagScale__array_set4_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_set5_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET5_F,
                  bHYPRE_StructDiagScale__array_set5_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_set6_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET6_F,
                  bHYPRE_StructDiagScale__array_set6_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_set7_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET7_F,
                  bHYPRE_StructDiagScale__array_set7_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_set_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SET_F,
                  bHYPRE_StructDiagScale__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_dimen_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_DIMEN_F,
                  bHYPRE_StructDiagScale__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_lower_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_LOWER_F,
                  bHYPRE_StructDiagScale__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_upper_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_UPPER_F,
                  bHYPRE_StructDiagScale__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_length_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_LENGTH_F,
                  bHYPRE_StructDiagScale__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_stride_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_STRIDE_F,
                  bHYPRE_StructDiagScale__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_iscolumnorder_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_StructDiagScale__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_isroworder_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_ISROWORDER_F,
                  bHYPRE_StructDiagScale__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_copy_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_COPY_F,
                  bHYPRE_StructDiagScale__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_smartcopy_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SMARTCOPY_F,
                  bHYPRE_StructDiagScale__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(bhypre_structdiagscale__array_slice_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_SLICE_F,
                  bHYPRE_StructDiagScale__array_slice_f)
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
SIDLFortran77Symbol(bhypre_structdiagscale__array_ensure_f,
                  BHYPRE_STRUCTDIAGSCALE__ARRAY_ENSURE_F,
                  bHYPRE_StructDiagScale__array_ensure_f)
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
static struct sidl_recursive_mutex_t bHYPRE_StructDiagScale__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_StructDiagScale__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_StructDiagScale__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_StructDiagScale__mutex )==EDEADLOCK) */
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

static struct bHYPRE_StructDiagScale__epv s_rem_epv__bhypre_structdiagscale;

static struct bHYPRE_Operator__epv s_rem_epv__bhypre_operator;

static struct bHYPRE_Solver__epv s_rem_epv__bhypre_solver;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_StructDiagScale__cast(
  struct bHYPRE_StructDiagScale__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "bHYPRE.StructDiagScale");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = self;
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.Solver");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_solver);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.Operator");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_operator);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.BaseClass");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
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
    cast =  (*func)(((struct 
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_StructDiagScale__delete(
  struct bHYPRE_StructDiagScale__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_StructDiagScale__getURL(
  struct bHYPRE_StructDiagScale__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_StructDiagScale__raddRef(
  struct bHYPRE_StructDiagScale__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
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
remote_bHYPRE_StructDiagScale__isRemote(
    struct bHYPRE_StructDiagScale__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_StructDiagScale__set_hooks(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale._set_hooks.", &throwaway_exception);
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
static void remote_bHYPRE_StructDiagScale__exec(
  struct bHYPRE_StructDiagScale__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_StructDiagScale_addRef(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_StructDiagScale__remote* r_obj = (struct 
      bHYPRE_StructDiagScale__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_StructDiagScale_deleteRef(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_StructDiagScale__remote* r_obj = (struct 
      bHYPRE_StructDiagScale__remote*)self->d_data;
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
remote_bHYPRE_StructDiagScale_isSame(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.isSame.", &throwaway_exception);
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
remote_bHYPRE_StructDiagScale_isType(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.isType.", &throwaway_exception);
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
remote_bHYPRE_StructDiagScale_getClassInfo(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.getClassInfo.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetOperator */
static int32_t
remote_bHYPRE_StructDiagScale_SetOperator(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ struct bHYPRE_Operator__object* A,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetOperator", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(A){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)A,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "A", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "A", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetOperator.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetTolerance */
static int32_t
remote_bHYPRE_StructDiagScale_SetTolerance(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ double tolerance,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetTolerance", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "tolerance", tolerance,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetTolerance.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetMaxIterations */
static int32_t
remote_bHYPRE_StructDiagScale_SetMaxIterations(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ int32_t max_iterations,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetMaxIterations", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "max_iterations", max_iterations,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetMaxIterations.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetLogging */
static int32_t
remote_bHYPRE_StructDiagScale_SetLogging(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ int32_t level,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetLogging", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "level", level, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetLogging.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetPrintLevel */
static int32_t
remote_bHYPRE_StructDiagScale_SetPrintLevel(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ int32_t level,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetPrintLevel", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "level", level, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetPrintLevel.", &throwaway_exception);
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

/* REMOTE METHOD STUB:GetNumIterations */
static int32_t
remote_bHYPRE_StructDiagScale_GetNumIterations(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* out */ int32_t* num_iterations,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetNumIterations", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.GetNumIterations.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackInt( _rsvp, "num_iterations", num_iterations,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetRelResidualNorm */
static int32_t
remote_bHYPRE_StructDiagScale_GetRelResidualNorm(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* out */ double* norm,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetRelResidualNorm", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.GetRelResidualNorm.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "norm", norm, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE_StructDiagScale_SetCommunicator(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetCommunicator.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetIntParameter */
static int32_t
remote_bHYPRE_StructDiagScale_SetIntParameter(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* in */ int32_t value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetIntParameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetIntParameter.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetDoubleParameter */
static int32_t
remote_bHYPRE_StructDiagScale_SetDoubleParameter(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetDoubleParameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetDoubleParameter.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetStringParameter */
static int32_t
remote_bHYPRE_StructDiagScale_SetStringParameter(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* in */ const char* value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetStringParameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetStringParameter.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetIntArray1Parameter */
static int32_t
remote_bHYPRE_StructDiagScale_SetIntArray1Parameter(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* in rarray[nvalues] */ struct sidl_int__array* value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetIntArray1Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "value", value,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetIntArray1Parameter.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetIntArray2Parameter */
static int32_t
remote_bHYPRE_StructDiagScale_SetIntArray2Parameter(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetIntArray2Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "value", value,
      sidl_column_major_order,2,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetIntArray2Parameter.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetDoubleArray1Parameter */
static int32_t
remote_bHYPRE_StructDiagScale_SetDoubleArray1Parameter(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* in rarray[nvalues] */ struct sidl_double__array* value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetDoubleArray1Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetDoubleArray1Parameter.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetDoubleArray2Parameter */
static int32_t
remote_bHYPRE_StructDiagScale_SetDoubleArray2Parameter(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetDoubleArray2Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,
      sidl_column_major_order,2,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.SetDoubleArray2Parameter.", &throwaway_exception);
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

/* REMOTE METHOD STUB:GetIntValue */
static int32_t
remote_bHYPRE_StructDiagScale_GetIntValue(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* out */ int32_t* value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetIntValue", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.GetIntValue.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetDoubleValue */
static int32_t
remote_bHYPRE_StructDiagScale_GetDoubleValue(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ const char* name,
  /* out */ double* value,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetDoubleValue", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.GetDoubleValue.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Setup */
static int32_t
remote_bHYPRE_StructDiagScale_Setup(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ struct bHYPRE_Vector__object* b,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Setup", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(b){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)b,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "b", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "b", NULL, _ex);SIDL_CHECK(*_ex);
    }
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.Setup.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Apply */
static int32_t
remote_bHYPRE_StructDiagScale_Apply(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ struct bHYPRE_Vector__object* b,
  /* inout */ struct bHYPRE_Vector__object** x,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Apply", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(b){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)b,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "b", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "b", NULL, _ex);SIDL_CHECK(*_ex);
    }
    if(*x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)*x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }
    /* Transfer this reference */
    if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
      SIDL_CHECK(*_ex);
      (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
        sidl_BaseInterface)*x)->d_object, _ex);SIDL_CHECK(*_ex);
      sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x,
        _ex);SIDL_CHECK(*_ex); 
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.Apply.", &throwaway_exception);
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

/* REMOTE METHOD STUB:ApplyAdjoint */
static int32_t
remote_bHYPRE_StructDiagScale_ApplyAdjoint(
  /* in */ struct bHYPRE_StructDiagScale__object* self ,
  /* in */ struct bHYPRE_Vector__object* b,
  /* inout */ struct bHYPRE_Vector__object** x,
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
      bHYPRE_StructDiagScale__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "ApplyAdjoint", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(b){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)b,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "b", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "b", NULL, _ex);SIDL_CHECK(*_ex);
    }
    if(*x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)*x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }
    /* Transfer this reference */
    if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
      SIDL_CHECK(*_ex);
      (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
        sidl_BaseInterface)*x)->d_object, _ex);SIDL_CHECK(*_ex);
      sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x,
        _ex);SIDL_CHECK(*_ex); 
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.StructDiagScale.ApplyAdjoint.", &throwaway_exception);
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

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_StructDiagScale__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_StructDiagScale__epv* epv = &s_rem_epv__bhypre_structdiagscale;
  struct bHYPRE_Operator__epv*        e0  = &s_rem_epv__bhypre_operator;
  struct bHYPRE_Solver__epv*          e1  = &s_rem_epv__bhypre_solver;
  struct sidl_BaseClass__epv*         e2  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*     e3  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast                         = remote_bHYPRE_StructDiagScale__cast;
  epv->f__delete                       = remote_bHYPRE_StructDiagScale__delete;
  epv->f__exec                         = remote_bHYPRE_StructDiagScale__exec;
  epv->f__getURL                       = remote_bHYPRE_StructDiagScale__getURL;
  epv->f__raddRef                      = remote_bHYPRE_StructDiagScale__raddRef;
  epv->f__isRemote                     = 
    remote_bHYPRE_StructDiagScale__isRemote;
  epv->f__set_hooks                    = 
    remote_bHYPRE_StructDiagScale__set_hooks;
  epv->f__ctor                         = NULL;
  epv->f__ctor2                        = NULL;
  epv->f__dtor                         = NULL;
  epv->f_addRef                        = remote_bHYPRE_StructDiagScale_addRef;
  epv->f_deleteRef                     = 
    remote_bHYPRE_StructDiagScale_deleteRef;
  epv->f_isSame                        = remote_bHYPRE_StructDiagScale_isSame;
  epv->f_isType                        = remote_bHYPRE_StructDiagScale_isType;
  epv->f_getClassInfo                  = 
    remote_bHYPRE_StructDiagScale_getClassInfo;
  epv->f_SetOperator                   = 
    remote_bHYPRE_StructDiagScale_SetOperator;
  epv->f_SetTolerance                  = 
    remote_bHYPRE_StructDiagScale_SetTolerance;
  epv->f_SetMaxIterations              = 
    remote_bHYPRE_StructDiagScale_SetMaxIterations;
  epv->f_SetLogging                    = 
    remote_bHYPRE_StructDiagScale_SetLogging;
  epv->f_SetPrintLevel                 = 
    remote_bHYPRE_StructDiagScale_SetPrintLevel;
  epv->f_GetNumIterations              = 
    remote_bHYPRE_StructDiagScale_GetNumIterations;
  epv->f_GetRelResidualNorm            = 
    remote_bHYPRE_StructDiagScale_GetRelResidualNorm;
  epv->f_SetCommunicator               = 
    remote_bHYPRE_StructDiagScale_SetCommunicator;
  epv->f_SetIntParameter               = 
    remote_bHYPRE_StructDiagScale_SetIntParameter;
  epv->f_SetDoubleParameter            = 
    remote_bHYPRE_StructDiagScale_SetDoubleParameter;
  epv->f_SetStringParameter            = 
    remote_bHYPRE_StructDiagScale_SetStringParameter;
  epv->f_SetIntArray1Parameter         = 
    remote_bHYPRE_StructDiagScale_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter         = 
    remote_bHYPRE_StructDiagScale_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter      = 
    remote_bHYPRE_StructDiagScale_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter      = 
    remote_bHYPRE_StructDiagScale_SetDoubleArray2Parameter;
  epv->f_GetIntValue                   = 
    remote_bHYPRE_StructDiagScale_GetIntValue;
  epv->f_GetDoubleValue                = 
    remote_bHYPRE_StructDiagScale_GetDoubleValue;
  epv->f_Setup                         = remote_bHYPRE_StructDiagScale_Setup;
  epv->f_Apply                         = remote_bHYPRE_StructDiagScale_Apply;
  epv->f_ApplyAdjoint                  = 
    remote_bHYPRE_StructDiagScale_ApplyAdjoint;

  e0->f__cast                    = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete                  = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL                  = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef                 = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote                = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks               = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec                    = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_SetCommunicator          = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e0->f_SetIntParameter          = (int32_t (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetIntParameter;
  e0->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,double,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleParameter;
  e0->f_SetStringParameter       = (int32_t (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_SetStringParameter;
  e0->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetIntArray1Parameter;
  e0->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetIntArray2Parameter;
  e0->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleArray1Parameter;
  e0->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleArray2Parameter;
  e0->f_GetIntValue              = (int32_t (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_GetIntValue;
  e0->f_GetDoubleValue           = (int32_t (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_GetDoubleValue;
  e0->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Setup;
  e0->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_Apply;
  e0->f_ApplyAdjoint             = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_ApplyAdjoint;
  e0->f_addRef                   = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef                = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType                   = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast                    = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete                  = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL                  = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef                 = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote                = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks               = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec                    = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_SetOperator              = (int32_t (*)(void*,
    struct bHYPRE_Operator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetOperator;
  e1->f_SetTolerance             = (int32_t (*)(void*,double,
    struct sidl_BaseInterface__object **)) epv->f_SetTolerance;
  e1->f_SetMaxIterations         = (int32_t (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetMaxIterations;
  e1->f_SetLogging               = (int32_t (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetLogging;
  e1->f_SetPrintLevel            = (int32_t (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetPrintLevel;
  e1->f_GetNumIterations         = (int32_t (*)(void*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_GetNumIterations;
  e1->f_GetRelResidualNorm       = (int32_t (*)(void*,double*,
    struct sidl_BaseInterface__object **)) epv->f_GetRelResidualNorm;
  e1->f_SetCommunicator          = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e1->f_SetIntParameter          = (int32_t (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetIntParameter;
  e1->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,double,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleParameter;
  e1->f_SetStringParameter       = (int32_t (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_SetStringParameter;
  e1->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetIntArray1Parameter;
  e1->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetIntArray2Parameter;
  e1->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleArray1Parameter;
  e1->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetDoubleArray2Parameter;
  e1->f_GetIntValue              = (int32_t (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_GetIntValue;
  e1->f_GetDoubleValue           = (int32_t (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_GetDoubleValue;
  e1->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Setup;
  e1->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_Apply;
  e1->f_ApplyAdjoint             = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_ApplyAdjoint;
  e1->f_addRef                   = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef                = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType                   = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e2->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete      = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_addRef       = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,
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

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_StructDiagScale__object*
bHYPRE_StructDiagScale__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_StructDiagScale__object* self;

  struct bHYPRE_StructDiagScale__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_StructDiagScale__remote* r_obj;
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
    return bHYPRE_StructDiagScale__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_StructDiagScale__object*) malloc(
      sizeof(struct bHYPRE_StructDiagScale__object));

  r_obj =
    (struct bHYPRE_StructDiagScale__remote*) malloc(
      sizeof(struct bHYPRE_StructDiagScale__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructDiagScale__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_solver.d_epv    = &s_rem_epv__bhypre_solver;
  s0->d_bhypre_solver.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_structdiagscale;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct bHYPRE_StructDiagScale__object*
bHYPRE_StructDiagScale__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_StructDiagScale__object* self;

  struct bHYPRE_StructDiagScale__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_StructDiagScale__remote* r_obj;
  self =
    (struct bHYPRE_StructDiagScale__object*) malloc(
      sizeof(struct bHYPRE_StructDiagScale__object));

  r_obj =
    (struct bHYPRE_StructDiagScale__remote*) malloc(
      sizeof(struct bHYPRE_StructDiagScale__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructDiagScale__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_solver.d_epv    = &s_rem_epv__bhypre_solver;
  s0->d_bhypre_solver.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_structdiagscale;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_StructDiagScale__object*
bHYPRE_StructDiagScale__remoteCreate(const char *url, sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct bHYPRE_StructDiagScale__object* self;

  struct bHYPRE_StructDiagScale__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_StructDiagScale__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.StructDiagScale",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_StructDiagScale__object*) malloc(
      sizeof(struct bHYPRE_StructDiagScale__object));

  r_obj =
    (struct bHYPRE_StructDiagScale__remote*) malloc(
      sizeof(struct bHYPRE_StructDiagScale__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructDiagScale__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_solver.d_epv    = &s_rem_epv__bhypre_solver;
  s0->d_bhypre_solver.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_structdiagscale;

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

struct bHYPRE_StructDiagScale__object*
bHYPRE_StructDiagScale__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_StructDiagScale__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.StructDiagScale",
      (void*)bHYPRE_StructDiagScale__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_StructDiagScale__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.StructDiagScale", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_StructDiagScale__object*
bHYPRE_StructDiagScale__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return bHYPRE_StructDiagScale__remoteConnect(url, ar, _ex);
}

