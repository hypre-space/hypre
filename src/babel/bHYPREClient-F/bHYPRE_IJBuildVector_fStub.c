/*
 * File:          bHYPRE_IJBuildVector_fStub.c
 * Symbol:        bHYPRE.IJBuildVector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:45 PST
 * Generated:     20030401 14:48:04 PST
 * Description:   Client-side glue code for bHYPRE.IJBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 203
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * Symbol "bHYPRE.IJBuildVector" (version 1.0.0)
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
#include "bHYPRE_IJBuildVector_IOR.h"
#include "SIDL_BaseInterface_IOR.h"
#include "SIDL_ClassInfo_IOR.h"

/*
 * Cast method for interface and type conversions.
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector__cast_f,BHYPRE_IJBUILDVECTOR__CAST_F,bHYPRE_IJBuildVector__cast_f)
(
  int64_t *ref,
  int64_t *retval
)
{
  struct SIDL_BaseInterface__object  *_base =
    (struct SIDL_BaseInterface__object *)(ptrdiff_t)*ref;
  if (_base) {
    *retval = (ptrdiff_t)(
      *_base->d_epv->f__cast)(
      _base->d_object,
      "bHYPRE.IJBuildVector");
  }
  else {
    *retval = 0;
  }
}

/*
 * Cast method for interface and class type conversions.
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector__cast2_f,BHYPRE_IJBUILDVECTOR__CAST2_F,bHYPRE_IJBuildVector__cast2_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  void* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f__cast))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(bhypre_ijbuildvector_addref_f,BHYPRE_IJBUILDVECTOR_ADDREF_F,bHYPRE_IJBuildVector_addRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_addRef))(
    _proxy_self->d_object
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
SIDLFortran77Symbol(bhypre_ijbuildvector_deleteref_f,BHYPRE_IJBUILDVECTOR_DELETEREF_F,bHYPRE_IJBuildVector_deleteRef_f)
(
  int64_t *self
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  (*(_epv->f_deleteRef))(
    _proxy_self->d_object
  );
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_issame_f,BHYPRE_IJBUILDVECTOR_ISSAME_F,bHYPRE_IJBuildVector_isSame_f)
(
  int64_t *self,
  int64_t *iobj,
  SIDL_F77_Bool *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_iobj = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _proxy_iobj =
    (struct SIDL_BaseInterface__object*)
    (ptrdiff_t)(*iobj);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isSame))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(bhypre_ijbuildvector_queryint_f,BHYPRE_IJBUILDVECTOR_QUERYINT_F,bHYPRE_IJBuildVector_queryInt_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  int64_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  struct SIDL_BaseInterface__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_queryInt))(
      _proxy_self->d_object,
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
SIDLFortran77Symbol(bhypre_ijbuildvector_istype_f,BHYPRE_IJBUILDVECTOR_ISTYPE_F,bHYPRE_IJBuildVector_isType_f)
(
  int64_t *self,
  SIDL_F77_String name
  SIDL_F77_STR_NEAR_LEN_DECL(name),
  SIDL_F77_Bool *retval
  SIDL_F77_STR_FAR_LEN_DECL(name)
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  char* _proxy_name = NULL;
  SIDL_bool _proxy_retval;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _proxy_name =
    SIDL_copy_fortran_str(SIDL_F77_STR(name),
      SIDL_F77_STR_LEN(name));
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_isType))(
      _proxy_self->d_object,
      _proxy_name
    );
  *retval = ((_proxy_retval == TRUE) ? SIDL_F77_TRUE : SIDL_F77_FALSE);
  free((void *)_proxy_name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_getclassinfo_f,BHYPRE_IJBUILDVECTOR_GETCLASSINFO_F,bHYPRE_IJBuildVector_getClassInfo_f)
(
  int64_t *self,
  int64_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  struct SIDL_ClassInfo__object* _proxy_retval = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  _proxy_retval = 
    (*(_epv->f_getClassInfo))(
      _proxy_self->d_object
    );
  *retval = (ptrdiff_t)_proxy_retval;
}

/*
 * Set the MPI Communicator.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_setcommunicator_f,BHYPRE_IJBUILDVECTOR_SETCOMMUNICATOR_F,bHYPRE_IJBuildVector_SetCommunicator_f)
(
  int64_t *self,
  int64_t *mpi_comm,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  void* _proxy_mpi_comm = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _proxy_mpi_comm =
    (void*)
    (ptrdiff_t)(*mpi_comm);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetCommunicator))(
      _proxy_self->d_object,
      _proxy_mpi_comm
    );
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_initialize_f,BHYPRE_IJBUILDVECTOR_INITIALIZE_F,bHYPRE_IJBuildVector_Initialize_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Initialize))(
      _proxy_self->d_object
    );
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_assemble_f,BHYPRE_IJBUILDVECTOR_ASSEMBLE_F,bHYPRE_IJBuildVector_Assemble_f)
(
  int64_t *self,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Assemble))(
      _proxy_self->d_object
    );
}

/*
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_getobject_f,BHYPRE_IJBUILDVECTOR_GETOBJECT_F,bHYPRE_IJBuildVector_GetObject_f)
(
  int64_t *self,
  int64_t *A,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  struct SIDL_BaseInterface__object* _proxy_A = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetObject))(
      _proxy_self->d_object,
      &_proxy_A
    );
  *A = (ptrdiff_t)_proxy_A;
}

/*
 * Set the local range for a vector object.  Each process owns
 * some unique consecutive range of vector unknowns, indicated
 * by the global indices {\tt jlower} and {\tt jupper}.  The
 * data is required to be such that the value of {\tt jlower} on
 * any process $p$ be exactly one more than the value of {\tt
 * jupper} on process $p-1$.  Note that the first index of the
 * global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_setlocalrange_f,BHYPRE_IJBUILDVECTOR_SETLOCALRANGE_F,bHYPRE_IJBuildVector_SetLocalRange_f)
(
  int64_t *self,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_SetLocalRange))(
      _proxy_self->d_object,
      *jlower,
      *jupper
    );
}

/*
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_setvalues_f,BHYPRE_IJBUILDVECTOR_SETVALUES_F,bHYPRE_IJBuildVector_SetValues_f)
(
  int64_t *self,
  int32_t *nvalues,
  int64_t *indices,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
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
      _proxy_self->d_object,
      *nvalues,
      _proxy_indices,
      _proxy_values
    );
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_addtovalues_f,BHYPRE_IJBUILDVECTOR_ADDTOVALUES_F,bHYPRE_IJBuildVector_AddToValues_f)
(
  int64_t *self,
  int32_t *nvalues,
  int64_t *indices,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
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
      _proxy_self->d_object,
      *nvalues,
      _proxy_indices,
      _proxy_values
    );
}

/*
 * Returns range of the part of the vector owned by this
 * processor.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_getlocalrange_f,BHYPRE_IJBUILDVECTOR_GETLOCALRANGE_F,bHYPRE_IJBuildVector_GetLocalRange_f)
(
  int64_t *self,
  int32_t *jlower,
  int32_t *jupper,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetLocalRange))(
      _proxy_self->d_object,
      jlower,
      jupper
    );
}

/*
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_getvalues_f,BHYPRE_IJBUILDVECTOR_GETVALUES_F,bHYPRE_IJBuildVector_GetValues_f)
(
  int64_t *self,
  int32_t *nvalues,
  int64_t *indices,
  int64_t *values,
  int32_t *retval
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  struct SIDL_int__array* _proxy_indices = NULL;
  struct SIDL_double__array* _proxy_values = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _proxy_indices =
    (struct SIDL_int__array*)
    (ptrdiff_t)(*indices);
  _proxy_values =
    (struct SIDL_double__array*)
    (ptrdiff_t)(*values);
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_GetValues))(
      _proxy_self->d_object,
      *nvalues,
      _proxy_indices,
      &_proxy_values
    );
  *values = (ptrdiff_t)_proxy_values;
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_print_f,BHYPRE_IJBUILDVECTOR_PRINT_F,bHYPRE_IJBuildVector_Print_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
    (ptrdiff_t)(*self);
  _proxy_filename =
    SIDL_copy_fortran_str(SIDL_F77_STR(filename),
      SIDL_F77_STR_LEN(filename));
  _epv = _proxy_self->d_epv;
  *retval = 
    (*(_epv->f_Print))(
      _proxy_self->d_object,
      _proxy_filename
    );
  free((void *)_proxy_filename);
}

/*
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */

void
SIDLFortran77Symbol(bhypre_ijbuildvector_read_f,BHYPRE_IJBUILDVECTOR_READ_F,bHYPRE_IJBuildVector_Read_f)
(
  int64_t *self,
  SIDL_F77_String filename
  SIDL_F77_STR_NEAR_LEN_DECL(filename),
  int64_t *comm,
  int32_t *retval
  SIDL_F77_STR_FAR_LEN_DECL(filename)
)
{
  struct bHYPRE_IJBuildVector__epv *_epv = NULL;
  struct bHYPRE_IJBuildVector__object* _proxy_self = NULL;
  char* _proxy_filename = NULL;
  void* _proxy_comm = NULL;
  _proxy_self =
    (struct bHYPRE_IJBuildVector__object*)
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
      _proxy_self->d_object,
      _proxy_filename,
      _proxy_comm
    );
  free((void *)_proxy_filename);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_createcol_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_CREATECOL_F,
                  bHYPRE_IJBuildVector__array_createCol_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createCol(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_createrow_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_CREATEROW_F,
                  bHYPRE_IJBuildVector__array_createRow_f)
  (int32_t *dimen,
   int32_t lower[],
   int32_t upper[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_createRow(*dimen, lower, upper);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_create1d_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_CREATE1D_F,
                  bHYPRE_IJBuildVector__array_create1d_f)
  (int32_t *len, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create1d(*len);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_create2dcol_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_CREATE2DCOL_F,
                  bHYPRE_IJBuildVector__array_create2dCol_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dCol(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_create2drow_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_CREATE2DROW_F,
                  bHYPRE_IJBuildVector__array_create2dRow_f)
  (int32_t *m, int32_t *n, int64_t *result)
{
  *result = (ptrdiff_t)SIDL_interface__array_create2dRow(*m, *n);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_addref_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_ADDREF_F,
                  bHYPRE_IJBuildVector__array_addRef_f)
  (int64_t *array)
{
  SIDL_interface__array_addRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_deleteref_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_DELETEREF_F,
                  bHYPRE_IJBuildVector__array_deleteRef_f)
  (int64_t *array)
{
  SIDL_interface__array_deleteRef((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_get1_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_GET1_F,
                  bHYPRE_IJBuildVector__array_get1_f)
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
SIDLFortran77Symbol(bhypre_ijbuildvector__array_get2_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_GET2_F,
                  bHYPRE_IJBuildVector__array_get2_f)
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
SIDLFortran77Symbol(bhypre_ijbuildvector__array_get3_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_GET3_F,
                  bHYPRE_IJBuildVector__array_get3_f)
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
SIDLFortran77Symbol(bhypre_ijbuildvector__array_get4_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_GET4_F,
                  bHYPRE_IJBuildVector__array_get4_f)
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
SIDLFortran77Symbol(bhypre_ijbuildvector__array_get_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_GET_F,
                  bHYPRE_IJBuildVector__array_get_f)
  (int64_t *array,
   int32_t indices[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_get((const struct SIDL_interface__array 
      *)(ptrdiff_t)*array, indices);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_set1_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_SET1_F,
                  bHYPRE_IJBuildVector__array_set1_f)
  (int64_t *array,
   int32_t *i1,
   int64_t *value)
{
  SIDL_interface__array_set1((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_set2_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_SET2_F,
                  bHYPRE_IJBuildVector__array_set2_f)
  (int64_t *array,
   int32_t *i1,
   int32_t *i2,
   int64_t *value)
{
  SIDL_interface__array_set2((struct SIDL_interface__array *)(ptrdiff_t)*array
  , *i1, *i2, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_set3_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_SET3_F,
                  bHYPRE_IJBuildVector__array_set3_f)
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
SIDLFortran77Symbol(bhypre_ijbuildvector__array_set4_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_SET4_F,
                  bHYPRE_IJBuildVector__array_set4_f)
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
SIDLFortran77Symbol(bhypre_ijbuildvector__array_set_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_SET_F,
                  bHYPRE_IJBuildVector__array_set_f)
  (int64_t *array,
  int32_t indices[],
  int64_t *value)
{
  SIDL_interface__array_set((struct SIDL_interface__array *)(ptrdiff_t)*array,
    indices, (struct SIDL_BaseInterface__object *)(ptrdiff_t)*value);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_dimen_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_DIMEN_F,
                  bHYPRE_IJBuildVector__array_dimen_f)
  (int64_t *array, int32_t *result)
{
  *result =
    SIDL_interface__array_dimen((struct SIDL_interface__array 
      *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_lower_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_LOWER_F,
                  bHYPRE_IJBuildVector__array_lower_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_lower((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_upper_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_UPPER_F,
                  bHYPRE_IJBuildVector__array_upper_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_upper((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_stride_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_STRIDE_F,
                  bHYPRE_IJBuildVector__array_stride_f)
  (int64_t *array,
   int32_t *ind,
   int32_t *result)
{
  *result = 
    SIDL_interface__array_stride((struct SIDL_interface__array 
      *)(ptrdiff_t)*array, *ind);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_iscolumnorder_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_ISCOLUMNORDER_F,
                  bHYPRE_IJBuildVector__array_isColumnOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isColumnOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_isroworder_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_ISROWORDER_F,
                  bHYPRE_IJBuildVector__array_isRowOrder_f)
  (int64_t *array,
   SIDL_F77_Bool *result)
{
  *result = SIDL_interface__array_isRowOrder((struct SIDL_interface__array 
    *)(ptrdiff_t)*array);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_copy_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_COPY_F,
                  bHYPRE_IJBuildVector__array_copy_f)
  (int64_t *src,
   int64_t *dest)
{
  SIDL_interface__array_copy((const struct SIDL_interface__array 
    *)(ptrdiff_t)*src,
                             (struct SIDL_interface__array *)(ptrdiff_t)*dest);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_smartcopy_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_SMARTCOPY_F,
                  bHYPRE_IJBuildVector__array_smartCopy_f)
  (int64_t *src)
{
  SIDL_interface__array_smartCopy((struct SIDL_interface__array 
    *)(ptrdiff_t)*src);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_slice_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_SLICE_F,
                  bHYPRE_IJBuildVector__array_slice_f)
  (int64_t *src,
   int32_t *dimen,
   int32_t numElem[],
   int32_t srcStart[],
   int32_t srcStride[],
   int32_t newStart[],
   int64_t *result)
{
  *result = (ptrdiff_t)
    SIDL_interface__array_slice((struct SIDL_interface__array *)(ptrdiff_t)*src,
      *dimen, numElem, srcStart, srcStride, newStart);
}

void
SIDLFortran77Symbol(bhypre_ijbuildvector__array_ensure_f,
                  BHYPRE_IJBUILDVECTOR__ARRAY_ENSURE_F,
                  bHYPRE_IJBuildVector__array_ensure_f)
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

