/*
 * File:          bHYPRE_ParCSRDiagScale_fStub.c
 * Symbol:        bHYPRE.ParCSRDiagScale-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.ParCSRDiagScale
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

/*
 * Symbol "bHYPRE.ParCSRDiagScale" (version 1.0.0)
 * 
 * Diagonal scaling preconditioner for ParCSR matrix class.
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 */

#include <stddef.h>
#include <stdlib.h>
#include "sidlfortran.h"
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stdio.h>
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "bHYPRE_ParCSRDiagScale_IOR.h"
#include "bHYPRE_MPICommunicator_IOR.h"
#include "bHYPRE_Operator_IOR.h"
#include "bHYPRE_IJParCSRMatrix_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "bHYPRE_Vector_IOR.h"
#include "sidl_BaseInterface_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct bHYPRE_ParCSRDiagScale__external* _getIOR(void)
{
  static const struct bHYPRE_ParCSRDiagScale__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = bHYPRE_ParCSRDiagScale__externals();
#else
    _ior = (struct 
      bHYPRE_ParCSRDiagScale__external*)sidl_dynamicLoadIOR(
      "bHYPRE.ParCSRDiagScale","bHYPRE_ParCSRDiagScale__externals") ;
#endif
  }
  return _ior;
}

/*
 * Return pointer to static functions.
 */

static const struct bHYPRE_ParCSRDiagScale__sepv* _getSEPV(void)
{
  static const struct bHYPRE_ParCSRDiagScale__sepv *_sepv = NULL;
  if (!_sepv) {
    _sepv = (*(_getIOR()->getStaticEPV))();
  }
  return _sepv;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__create_f,BHYPRE_PARCSRDIAGSCALE__CREATE_F,bHYPRE_ParCSRDiagScale__create_f)
(
  int64_t *self
)
{
  *self = (ptrdiff_t) (*(_getIOR()->createObject))();
}

/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__cast_f,BHYPRE_PARCSRDIAGSCALE__CAST_F,bHYPRE_ParCSRDiagScale__cast_f)
(
  int64_t *ref,
  int64_t *retval
)
{
  struct sidl_BaseInterface__object  *_base =
    (struct sidl_BaseInterface__object *)(ptrdiff_t)*ref;
  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.ParCSRDiagScale");
  } else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__cast2_f,BHYPRE_PARCSRDIAGSCALE__CAST2_F,bHYPRE_ParCSRDiagScale__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale_addref_f,BHYPRE_PARCSRDIAGSCALE_ADDREF_F,bHYPRE_ParCSRDiagScale_addRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_deleteref_f,BHYPRE_PARCSRDIAGSCALE_DELETEREF_F,bHYPRE_ParCSRDiagScale_deleteRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_issame_f,BHYPRE_PARCSRDIAGSCALE_ISSAME_F,bHYPRE_ParCSRDiagScale_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  struct sidl_BaseInterface__object* _proxy_iobj = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct sidl_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self,
      _proxy_iobj
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_queryint_f,BHYPRE_PARCSRDIAGSCALE_QUERYINT_F,bHYPRE_ParCSRDiagScale_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self,
      _proxy_name
    );
  *retval = (ptrdiff_t)_proxy_retval;
  free((void *)_proxy_name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_istype_f,BHYPRE_PARCSRDIAGSCALE_ISTYPE_F,bHYPRE_ParCSRDiagScale_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  sidl_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_getclassinfo_f,BHYPRE_PARCSRDIAGSCALE_GETCLASSINFO_F,bHYPRE_ParCSRDiagScale_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  struct sidl_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Method:  Create[]
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_create_f,BHYPRE_PARCSRDIAGSCALE_CREATE_F,bHYPRE_ParCSRDiagScale_Create_f)
(
  int64_t *mpi_comm,
  int64_t *A,
  int64_t *retval
)
{
  const struct bHYPRE_ParCSRDiagScale__sepv *_epv = _getSEPV();
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  struct bHYPRE_IJParCSRMatrix__object* _proxy_A = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_retval = NULL;
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _proxy_A =
    (struct bHYPRE_IJParCSRMatrix__object*)
    (ptrdiff_t)(*A);
  _proxy_retval = 
    (*(_epv->f_Create))(
      _proxy_mpi_comm,
      _proxy_A
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setcommunicator_f,BHYPRE_PARCSRDIAGSCALE_SETCOMMUNICATOR_F,bHYPRE_ParCSRDiagScale_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_MPICommunicator__object* _proxy_mpi_comm = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (struct bHYPRE_MPICommunicator__object*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self,
      _proxy_mpi_comm
    );
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setintparameter_f,BHYPRE_PARCSRDIAGSCALE_SETINTPARAMETER_F,bHYPRE_ParCSRDiagScale_SetIntParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetIntParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setdoubleparameter_f,BHYPRE_PARCSRDIAGSCALE_SETDOUBLEPARAMETER_F,bHYPRE_ParCSRDiagScale_SetDoubleParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetDoubleParameter))(
      _proxy_self,
      _proxy_name,
      *value
    );
  free((void *)_proxy_name);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setstringparameter_f,BHYPRE_PARCSRDIAGSCALE_SETSTRINGPARAMETER_F,bHYPRE_ParCSRDiagScale_SetStringParameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_String value
  SIDL_F77_STR_NEAR_LEN_DECL(value),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
  SIDL_F77_STR_FAR_LEN_DECL(value)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  char* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      _proxy_value
    );
  free((void *)_proxy_name);
  free((void *)_proxy_value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setintarray1parameter_f,BHYPRE_PARCSRDIAGSCALE_SETINTARRAY1PARAMETER_F,bHYPRE_ParCSRDiagScale_SetIntArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *nvalues,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array _alt_value;
  struct sidl_int__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setintarray2parameter_f,BHYPRE_PARCSRDIAGSCALE_SETINTARRAY2PARAMETER_F,bHYPRE_ParCSRDiagScale_SetIntArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_int__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setdoublearray1parameter_f,BHYPRE_PARCSRDIAGSCALE_SETDOUBLEARRAY1PARAMETER_F,bHYPRE_ParCSRDiagScale_SetDoubleArray1Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *nvalues,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array _alt_value;
  struct sidl_double__array* _proxy_value = &_alt_value;
  int32_t value_lower[1], value_upper[1], value_stride[1];
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setdoublearray2parameter_f,BHYPRE_PARCSRDIAGSCALE_SETDOUBLEARRAY2PARAMETER_F,bHYPRE_ParCSRDiagScale_SetDoubleArray2Parameter_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct sidl_double__array* _proxy_value = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      _proxy_value
    );
  free((void *)_proxy_name);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_getintvalue_f,BHYPRE_PARCSRDIAGSCALE_GETINTVALUE_F,bHYPRE_ParCSRDiagScale_GetIntValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int32_t *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetIntValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_getdoublevalue_f,BHYPRE_PARCSRDIAGSCALE_GETDOUBLEVALUE_F,bHYPRE_ParCSRDiagScale_GetDoubleValue_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  double *value,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    sidl_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetDoubleValue))(
      _proxy_self,
      _proxy_name,
      value
    );
  free((void *)_proxy_name);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setup_f,BHYPRE_PARCSRDIAGSCALE_SETUP_F,bHYPRE_ParCSRDiagScale_Setup_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      _proxy_x
    );
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_apply_f,BHYPRE_PARCSRDIAGSCALE_APPLY_F,bHYPRE_ParCSRDiagScale_Apply_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      &_proxy_x
    );
  *x = (ptrdiff_t)_proxy_x;
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_applyadjoint_f,BHYPRE_PARCSRDIAGSCALE_APPLYADJOINT_F,bHYPRE_ParCSRDiagScale_ApplyAdjoint_f)
(
  int64_t *self,
  int64_t *b,
  int64_t *x,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Vector__object* _proxy_b = NULL;
  struct bHYPRE_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
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
      &_proxy_x
    );
  *x = (ptrdiff_t)_proxy_x;
}

/*
 * Set the operator for the linear system being solved.
 * DEPRECATED.  use Create
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setoperator_f,BHYPRE_PARCSRDIAGSCALE_SETOPERATOR_F,bHYPRE_ParCSRDiagScale_SetOperator_f)
(
  int64_t *self,
  int64_t *A,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  struct bHYPRE_Operator__object* _proxy_A = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _proxy_A =
    (struct bHYPRE_Operator__object*)
    (ptrdiff_t)(*A);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetOperator))(
      _proxy_self,
      _proxy_A
    );
}

/*
 * (Optional) Set the convergence tolerance.
 * DEPRECATED.  use SetDoubleParameter
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_settolerance_f,BHYPRE_PARCSRDIAGSCALE_SETTOLERANCE_F,bHYPRE_ParCSRDiagScale_SetTolerance_f)
(
  int64_t *self,
  double *tolerance,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetTolerance))(
      _proxy_self,
      *tolerance
    );
}

/*
 * (Optional) Set maximum number of iterations.
 * DEPRECATED   use SetIntParameter
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setmaxiterations_f,BHYPRE_PARCSRDIAGSCALE_SETMAXITERATIONS_F,bHYPRE_ParCSRDiagScale_SetMaxIterations_f)
(
  int64_t *self,
  int32_t *max_iterations,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetMaxIterations))(
      _proxy_self,
      *max_iterations
    );
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setlogging_f,BHYPRE_PARCSRDIAGSCALE_SETLOGGING_F,bHYPRE_ParCSRDiagScale_SetLogging_f)
(
  int64_t *self,
  int32_t *level,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLogging))(
      _proxy_self,
      *level
    );
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * DEPRECATED   use SetIntParameter
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_setprintlevel_f,BHYPRE_PARCSRDIAGSCALE_SETPRINTLEVEL_F,bHYPRE_ParCSRDiagScale_SetPrintLevel_f)
(
  int64_t *self,
  int32_t *level,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetPrintLevel))(
      _proxy_self,
      *level
    );
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_getnumiterations_f,BHYPRE_PARCSRDIAGSCALE_GETNUMITERATIONS_F,bHYPRE_ParCSRDiagScale_GetNumIterations_f)
(
  int64_t *self,
  int32_t *num_iterations,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetNumIterations))(
      _proxy_self,
      num_iterations
    );
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale_getrelresidualnorm_f,BHYPRE_PARCSRDIAGSCALE_GETRELRESIDUALNORM_F,bHYPRE_ParCSRDiagScale_GetRelResidualNorm_f)
(
  int64_t *self,
  double *norm,
  int32_t *retval
)
{
  struct bHYPRE_ParCSRDiagScale__epv *_epv = NULL;
  struct bHYPRE_ParCSRDiagScale__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_ParCSRDiagScale__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRelResidualNorm))(
      _proxy_self,
      norm
    );
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_createcol_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_CREATECOL_F,
                  bHYPRE_ParCSRDiagScale__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_createrow_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_CREATEROW_F,
                  bHYPRE_ParCSRDiagScale__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_create1d_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_CREATE1D_F,
                  bHYPRE_ParCSRDiagScale__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_create2dcol_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_CREATE2DCOL_F,
                  bHYPRE_ParCSRDiagScale__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_create2drow_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_CREATE2DROW_F,
                  bHYPRE_ParCSRDiagScale__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)sidl_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_addref_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_ADDREF_F,
                  bHYPRE_ParCSRDiagScale__array_addRef_f)
  (int64_t *array)
{
  sidl_interface__array_addRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_deleteref_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_DELETEREF_F,
                  bHYPRE_ParCSRDiagScale__array_deleteRef_f)
  (int64_t *array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get1_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET1_F,
                  bHYPRE_ParCSRDiagScale__array_get1_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get2_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET2_F,
                  bHYPRE_ParCSRDiagScale__array_get2_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get3_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET3_F,
                  bHYPRE_ParCSRDiagScale__array_get3_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get4_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET4_F,
                  bHYPRE_ParCSRDiagScale__array_get4_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get5_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET5_F,
                  bHYPRE_ParCSRDiagScale__array_get5_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get6_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET6_F,
                  bHYPRE_ParCSRDiagScale__array_get6_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get7_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET7_F,
                  bHYPRE_ParCSRDiagScale__array_get7_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_get_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_GET_F,
                  bHYPRE_ParCSRDiagScale__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    sidl_interface__array_get((const struct sidl_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set1_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET1_F,
                  bHYPRE_ParCSRDiagScale__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set2_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET2_F,
                  bHYPRE_ParCSRDiagScale__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set3_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET3_F,
                  bHYPRE_ParCSRDiagScale__array_set3_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set4_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET4_F,
                  bHYPRE_ParCSRDiagScale__array_set4_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set5_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET5_F,
                  bHYPRE_ParCSRDiagScale__array_set5_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set6_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET6_F,
                  bHYPRE_ParCSRDiagScale__array_set6_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set7_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET7_F,
                  bHYPRE_ParCSRDiagScale__array_set7_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_set_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SET_F,
                  bHYPRE_ParCSRDiagScale__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  sidl_interface__array_set((struct sidl_interface__array *)(ptrdiff_t)*array,
    indices, (struct sidl_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_dimen_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_DIMEN_F,
                  bHYPRE_ParCSRDiagScale__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    sidl_interface__array_dimen((struct sidl_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_lower_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_LOWER_F,
                  bHYPRE_ParCSRDiagScale__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_lower((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_upper_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_UPPER_F,
                  bHYPRE_ParCSRDiagScale__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_upper((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_length_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_LENGTH_F,
                  bHYPRE_ParCSRDiagScale__array_length_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_length((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_stride_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_STRIDE_F,
                  bHYPRE_ParCSRDiagScale__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    sidl_interface__array_stride((struct sidl_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_iscolumnorder_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_ParCSRDiagScale__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_isroworder_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_ISROWORDER_F,
                  bHYPRE_ParCSRDiagScale__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_copy_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_COPY_F,
                  bHYPRE_ParCSRDiagScale__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array 
    *)(ptrdiff_t)*src,
                             (struct sidl_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_smartcopy_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SMARTCOPY_F,
                  bHYPRE_ParCSRDiagScale__array_smartCopy_f)
  (int64_t *src)
{
  sidl_interface__array_smartCopy((struct sidl_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_slice_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_SLICE_F,
                  bHYPRE_ParCSRDiagScale__array_slice_f)
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
SIDLFortran77Symbol(bhypre_parcsrdiagscale__array_ensure_f,
                  BHYPRE_PARCSRDIAGSCALE__ARRAY_ENSURE_F,
                  bHYPRE_ParCSRDiagScale__array_ensure_f)
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

