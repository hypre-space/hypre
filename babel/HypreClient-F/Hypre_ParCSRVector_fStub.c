/*
 * File:          Hypre_ParCSRVector_fStub.c
 * Symbol:        Hypre.ParCSRVector-v0.1.6
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:22 PST
 * Generated:     20030121 14:39:30 PST
 * Description:   Client-side glue code for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 435
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * Symbol "Hypre.ParCSRVector" (version 0.1.6)
 */

#include <stddef.h>
#include <stdlib.h>
#include "SIDLfortran.h"
#include "SIDL_header.h"
#ifndef included_SIDL_interface_IOR_h
#include "SIDL_interface_IOR.h"
#endif
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include "SIDL_Loader.h"
#endif
#include "Hypre_ParCSRVector_IOR.h"
#include "Hypre_Vector_IOR.h"
#include "SIDL_BaseInterface_IOR.h"
#include "SIDL_ClassInfo_IOR.h"

/*
 * Return pointer to internal IOR functions.
 */

static const struct Hypre_ParCSRVector__external* _getIOR(void)
{
  static const struct Hypre_ParCSRVector__external *_ior = NULL;
  if (!_ior) {
#ifdef SIDL_STATIC_LIBRARY
    _ior = Hypre_ParCSRVector__externals();
#else
    const struct Hypre_ParCSRVector__external*(*dll_f)(void) =
      (const struct Hypre_ParCSRVector__external*(*)(void)) 
        SIDL_Loader_lookupSymbol(
        "Hypre_ParCSRVector__externals");
    _ior = (dll_f ? (*dll_f)() : NULL);
    if (!_ior) {
      fputs("Unable to find the implementation for Hypre.ParCSRVector; please set SIDL_DLL_PATH\n", stderr);
      exit(-1);
    }
#endif
  }
  return _ior;
}

/*
 * Constructor for the class.
 */

void
SIDLFortran77Symbol(hypre_parcsrvector__create_f,HYPRE_PARCSRVECTOR__CREATE_F,Hypre_ParCSRVector__create_f)
(
  int64_t *self
)
{
  *self = (ptrdiff_t) (*(_getIOR()->createObject))();
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(hypre_parcsrvector__cast_f,HYPRE_PARCSRVECTOR__CAST_F,Hypre_ParCSRVector__cast_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
 * Object in <code>SIDL</code> have an intrinsic reference count.
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
SIDLFortran77Symbol(hypre_parcsrvector_addref_f,HYPRE_PARCSRVECTOR_ADDREF_F,Hypre_ParCSRVector_addRef_f)
(
  int64_t *self
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self
  );
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_deleteref_f,HYPRE_PARCSRVECTOR_DELETEREF_F,Hypre_ParCSRVector_deleteRef_f)
(
  int64_t *self
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
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
SIDLFortran77Symbol(hypre_parcsrvector_issame_f,HYPRE_PARCSRVECTOR_ISSAME_F,Hypre_ParCSRVector_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct SIDL_BaseInterface__object*)
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
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_queryint_f,HYPRE_PARCSRVECTOR_QUERYINT_F,Hypre_ParCSRVector_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_istype_f,HYPRE_PARCSRVECTOR_ISTYPE_F,Hypre_ParCSRVector_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
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
SIDLFortran77Symbol(hypre_parcsrvector_getclassinfo_f,HYPRE_PARCSRVECTOR_GETCLASSINFO_F,Hypre_ParCSRVector_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Method:  SetCommunicator[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_setcommunicator_f,HYPRE_PARCSRVECTOR_SETCOMMUNICATOR_F,Hypre_ParCSRVector_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  void* _proxy_mpi_comm = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (void*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self,
      _proxy_mpi_comm
    );
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_initialize_f,HYPRE_PARCSRVECTOR_INITIALIZE_F,Hypre_ParCSRVector_Initialize_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Initialize))(
      _proxy_self
    );
}

/*
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_assemble_f,HYPRE_PARCSRVECTOR_ASSEMBLE_F,Hypre_ParCSRVector_Assemble_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Assemble))(
      _proxy_self
    );
}

/*
 * The problem definition interface is a "builder" that creates an object
 * that contains the problem definition information, e.g. a matrix. To
 * perform subsequent operations with that object, it must be returned from
 * the problem definition object. "GetObject" performs this function.
 * <note>At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface. QueryInterface or Cast must
 * be used on the returned object to convert it into a known type.</note>
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_getobject_f,HYPRE_PARCSRVECTOR_GETOBJECT_F,Hypre_ParCSRVector_GetObject_f)
(
  int64_t *self,
  int64_t *A,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_A = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetObject))(
      _proxy_self,
      &_proxy_A
    );
  *A = (ptrdiff_t)_proxy_A;
}

/*
 * Method:  SetGlobalSize[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_setglobalsize_f,HYPRE_PARCSRVECTOR_SETGLOBALSIZE_F,Hypre_ParCSRVector_SetGlobalSize_f)
(
  int64_t *self,
  int32_t *n,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetGlobalSize))(
      _proxy_self,
      *n
    );
}

/*
 * Method:  SetPartitioning[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_setpartitioning_f,HYPRE_PARCSRVECTOR_SETPARTITIONING_F,Hypre_ParCSRVector_SetPartitioning_f)
(
  int64_t *self,
  int64_t *partitioning,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_partitioning = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_partitioning =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*partitioning);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetPartitioning))(
      _proxy_self,
      _proxy_partitioning
    );
}

/*
 * Method:  SetLocalComponents[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_setlocalcomponents_f,HYPRE_PARCSRVECTOR_SETLOCALCOMPONENTS_F,Hypre_ParCSRVector_SetLocalComponents_f)
(
  int64_t *self,
  int32_t *num_values,
  int64_t *glob_vec_indices,
  int64_t *value_indices,
  int64_t *values,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_glob_vec_indices = NULL;
  struct SIDL_int__array* _proxy_value_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_glob_vec_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*glob_vec_indices);
  _proxy_value_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*value_indices);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLocalComponents))(
      _proxy_self,
      *num_values,
      _proxy_glob_vec_indices,
      _proxy_value_indices,
      _proxy_values
    );
}

/*
 * Method:  AddtoLocalComponents[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_addtolocalcomponents_f,HYPRE_PARCSRVECTOR_ADDTOLOCALCOMPONENTS_F,Hypre_ParCSRVector_AddtoLocalComponents_f)
(
  int64_t *self,
  int32_t *num_values,
  int64_t *glob_vec_indices,
  int64_t *value_indices,
  int64_t *values,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_glob_vec_indices = NULL;
  struct SIDL_int__array* _proxy_value_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_glob_vec_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*glob_vec_indices);
  _proxy_value_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*value_indices);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddtoLocalComponents))(
      _proxy_self,
      *num_values,
      _proxy_glob_vec_indices,
      _proxy_value_indices,
      _proxy_values
    );
}

/*
 * Method:  SetLocalComponentsInBlock[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_setlocalcomponentsinblock_f,HYPRE_PARCSRVECTOR_SETLOCALCOMPONENTSINBLOCK_F,Hypre_ParCSRVector_SetLocalComponentsInBlock_f)
(
  int64_t *self,
  int32_t *glob_vec_index_start,
  int32_t *glob_vec_index_stop,
  int64_t *value_indices,
  int64_t *values,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_value_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_value_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*value_indices);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLocalComponentsInBlock))(
      _proxy_self,
      *glob_vec_index_start,
      *glob_vec_index_stop,
      _proxy_value_indices,
      _proxy_values
    );
}

/*
 * Method:  AddToLocalComponentsInBlock[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_addtolocalcomponentsinblock_f,HYPRE_PARCSRVECTOR_ADDTOLOCALCOMPONENTSINBLOCK_F,Hypre_ParCSRVector_AddToLocalComponentsInBlock_f)
(
  int64_t *self,
  int32_t *glob_vec_index_start,
  int32_t *glob_vec_index_stop,
  int64_t *value_indices,
  int64_t *values,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_value_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_value_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*value_indices);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddToLocalComponentsInBlock))(
      _proxy_self,
      *glob_vec_index_start,
      *glob_vec_index_stop,
      _proxy_value_indices,
      _proxy_values
    );
}

/*
 * Create a vector object.  Each process owns some unique consecutive
 * range of vector unknowns, indicated by the global indices {\tt
 * jlower} and {\tt jupper}.  The data is required to be such that the
 * value of {\tt jlower} on any process $p$ be exactly one more than
 * the value of {\tt jupper} on process $p-1$.  Note that the first
 * index of the global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_create_f,HYPRE_PARCSRVECTOR_CREATE_F,Hypre_ParCSRVector_Create_f)
(
  int64_t *self,
  int64_t *comm,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  void* _proxy_comm = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_comm =
    (void*)
    (ptrdiff_t)(*comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Create))(
      _proxy_self,
      _proxy_comm,
      *jlower,
      *jupper
    );
}

/*
 * Sets values in vector.  The arrays {\tt values} and {\tt indices}
 * are of dimension {\tt nvalues} and contain the vector values to be
 * set and the corresponding global vector indices, respectively.
 * Erases any previous values at the specified locations and replaces
 * them with new ones.
 * 
 * Not collective.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_setvalues_f,HYPRE_PARCSRVECTOR_SETVALUES_F,Hypre_ParCSRVector_SetValues_f)
(
  int64_t *self,
  int32_t *nvalues,
  int64_t *indices,
  int64_t *values,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*indices);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetValues))(
      _proxy_self,
      *nvalues,
      _proxy_indices,
      _proxy_values
    );
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{SetValues}.
 * 
 * Not collective.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_addtovalues_f,HYPRE_PARCSRVECTOR_ADDTOVALUES_F,Hypre_ParCSRVector_AddToValues_f)
(
  int64_t *self,
  int32_t *nvalues,
  int64_t *indices,
  int64_t *values,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*indices);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_AddToValues))(
      _proxy_self,
      *nvalues,
      _proxy_indices,
      _proxy_values
    );
}

/*
 * Read the vector from file.  This is mainly for debugging purposes.
 * 
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_read_f,HYPRE_PARCSRVECTOR_READ_F,Hypre_ParCSRVector_Read_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int64_t *comm,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  void* _proxy_comm = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    SIDL_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _proxy_comm =
    (void*)
    (ptrdiff_t)(*comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Read))(
      _proxy_self,
      _proxy_filename,
      _proxy_comm
    );
  free((void *)_proxy_filename);
}

/*
 * Print the vector to file.  This is mainly for debugging purposes.
 * 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_print_f,HYPRE_PARCSRVECTOR_PRINT_F,Hypre_ParCSRVector_Print_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    SIDL_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Print))(
      _proxy_self,
      _proxy_filename
    );
  free((void *)_proxy_filename);
}

/*
 * Method:  GetRow[]
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_getrow_f,HYPRE_PARCSRVECTOR_GETROW_F,Hypre_ParCSRVector_GetRow_f)
(
  int64_t *self,
  int32_t *row,
  int32_t *size,
  int64_t *col_ind,
  int64_t *values,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_col_ind = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetRow))(
      _proxy_self,
      *row,
      size,
      &_proxy_col_ind,
      &_proxy_values
    );
  *col_ind = (ptrdiff_t)_proxy_col_ind;
  *values = (ptrdiff_t)_proxy_values;
}

/*
 * y <- 0 (where y=self)
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_clear_f,HYPRE_PARCSRVECTOR_CLEAR_F,Hypre_ParCSRVector_Clear_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Clear))(
      _proxy_self
    );
}

/*
 * y <- x 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_copy_f,HYPRE_PARCSRVECTOR_COPY_F,Hypre_ParCSRVector_Copy_f)
(
  int64_t *self,
  int64_t *x,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct Hypre_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_x =
    (struct Hypre_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Copy))(
      _proxy_self,
      _proxy_x
    );
}

/*
 * create an x compatible with y
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_clone_f,HYPRE_PARCSRVECTOR_CLONE_F,Hypre_ParCSRVector_Clone_f)
(
  int64_t *self,
  int64_t *x,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct Hypre_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Clone))(
      _proxy_self,
      &_proxy_x
    );
  *x = (ptrdiff_t)_proxy_x;
}

/*
 * y <- a*y 
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_scale_f,HYPRE_PARCSRVECTOR_SCALE_F,Hypre_ParCSRVector_Scale_f)
(
  int64_t *self,
  double *a,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Scale))(
      _proxy_self,
      *a
    );
}

/*
 * d <- (y,x)
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_dot_f,HYPRE_PARCSRVECTOR_DOT_F,Hypre_ParCSRVector_Dot_f)
(
  int64_t *self,
  int64_t *x,
  double *d,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct Hypre_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_x =
    (struct Hypre_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Dot))(
      _proxy_self,
      _proxy_x,
      d
    );
}

/*
 * y <- a*x + y
 */

void
SIDLFortran77Symbol(hypre_parcsrvector_axpy_f,HYPRE_PARCSRVECTOR_AXPY_F,Hypre_ParCSRVector_Axpy_f)
(
  int64_t *self,
  double *a,
  int64_t *x,
  int32_t *retval
)
{
  struct Hypre_ParCSRVector__epv *_epv = NULL;
  struct Hypre_ParCSRVector__object* _proxy_self = NULL;
  struct Hypre_Vector__object* _proxy_x = NULL;
  _proxy_self =
    (struct Hypre_ParCSRVector__object*)
    (ptrdiff_t)(*self);
  _proxy_x =
    (struct Hypre_Vector__object*)
    (ptrdiff_t)(*x);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Axpy))(
      _proxy_self,
      *a,
      _proxy_x
    );
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_createcol_f,
                  HYPRE_PARCSRVECTOR__ARRAY_CREATECOL_F,
                  Hypre_ParCSRVector__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_createrow_f,
                  HYPRE_PARCSRVECTOR__ARRAY_CREATEROW_F,
                  Hypre_ParCSRVector__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_create1d_f,
                  HYPRE_PARCSRVECTOR__ARRAY_CREATE1D_F,
                  Hypre_ParCSRVector__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_create2dcol_f,
                  HYPRE_PARCSRVECTOR__ARRAY_CREATE2DCOL_F,
                  Hypre_ParCSRVector__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_create2drow_f,
                  HYPRE_PARCSRVECTOR__ARRAY_CREATE2DROW_F,
                  Hypre_ParCSRVector__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_addref_f,
                  HYPRE_PARCSRVECTOR__ARRAY_ADDREF_F,
                  Hypre_ParCSRVector__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_deleteref_f,
                  HYPRE_PARCSRVECTOR__ARRAY_DELETEREF_F,
                  Hypre_ParCSRVector__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_get1_f,
                  HYPRE_PARCSRVECTOR__ARRAY_GET1_F,
                  Hypre_ParCSRVector__array_get1_f)
  (int64_t *array, 
   int32_t *i1, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get1((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_get2_f,
                  HYPRE_PARCSRVECTOR__ARRAY_GET2_F,
                  Hypre_ParCSRVector__array_get2_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get2((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_get3_f,
                  HYPRE_PARCSRVECTOR__ARRAY_GET3_F,
                  Hypre_ParCSRVector__array_get3_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get3((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_get4_f,
                  HYPRE_PARCSRVECTOR__ARRAY_GET4_F,
                  Hypre_ParCSRVector__array_get4_f)
  (int64_t *array, 
   int32_t *i1, 
   int32_t *i2, 
   int32_t *i3, 
   int32_t *i4, 
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get4((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array
    , *i1, *i2, *i3, *i4);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_get_f,
                  HYPRE_PARCSRVECTOR__ARRAY_GET_F,
                  Hypre_ParCSRVector__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_set1_f,
                  HYPRE_PARCSRVECTOR__ARRAY_SET1_F,
                  Hypre_ParCSRVector__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_set2_f,
                  HYPRE_PARCSRVECTOR__ARRAY_SET2_F,
                  Hypre_ParCSRVector__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_set3_f,
                  HYPRE_PARCSRVECTOR__ARRAY_SET3_F,
                  Hypre_ParCSRVector__array_set3_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int64_t *value)
{
  SIDL_interface__array_set3((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_set4_f,
                  HYPRE_PARCSRVECTOR__ARRAY_SET4_F,
                  Hypre_ParCSRVector__array_set4_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int32_t *i3,
   int32_t *i4,
   int64_t *value)
{
  SIDL_interface__array_set4((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, *i3, *i4, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_set_f,
                  HYPRE_PARCSRVECTOR__ARRAY_SET_F,
                  Hypre_ParCSRVector__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_dimen_f,
                  HYPRE_PARCSRVECTOR__ARRAY_DIMEN_F,
                  Hypre_ParCSRVector__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_lower_f,
                  HYPRE_PARCSRVECTOR__ARRAY_LOWER_F,
                  Hypre_ParCSRVector__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_lower((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_upper_f,
                  HYPRE_PARCSRVECTOR__ARRAY_UPPER_F,
                  Hypre_ParCSRVector__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_upper((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_stride_f,
                  HYPRE_PARCSRVECTOR__ARRAY_STRIDE_F,
                  Hypre_ParCSRVector__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_iscolumnorder_f,
                  HYPRE_PARCSRVECTOR__ARRAY_ISCOLUMNORDER_F,
                  Hypre_ParCSRVector__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_isroworder_f,
                  HYPRE_PARCSRVECTOR__ARRAY_ISROWORDER_F,
                  Hypre_ParCSRVector__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_copy_f,
                  HYPRE_PARCSRVECTOR__ARRAY_COPY_F,
                  Hypre_ParCSRVector__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_smartcopy_f,
                  HYPRE_PARCSRVECTOR__ARRAY_SMARTCOPY_F,
                  Hypre_ParCSRVector__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(hypre_parcsrvector__array_ensure_f,
                  HYPRE_PARCSRVECTOR__ARRAY_ENSURE_F,
                  Hypre_ParCSRVector__array_ensure_f)
  (int64_t *src,
   int32_t *dimen,
   int     *ordering,
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_ensure((struct SIDL_interface__array 
      *)(ptrdiff_t)*src,
    *dimen, *ordering);
}

