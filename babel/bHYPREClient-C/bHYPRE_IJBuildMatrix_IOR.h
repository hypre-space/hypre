/*
 * File:          bHYPRE_IJBuildMatrix_IOR.h
 * Symbol:        bHYPRE.IJBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:45 PST
 * Generated:     20050317 11:17:47 PST
 * Description:   Intermediate Object Representation for bHYPRE.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 85
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJBuildMatrix_IOR_h
#define included_bHYPRE_IJBuildMatrix_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.IJBuildMatrix" (version 1.0.0)
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 * 
 */

struct bHYPRE_IJBuildMatrix__array;
struct bHYPRE_IJBuildMatrix__object;

extern struct bHYPRE_IJBuildMatrix__object*
bHYPRE_IJBuildMatrix__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_IJBuildMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  sidl_bool (*f_isSame)(
    void* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  sidl_bool (*f_isType)(
    void* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    void* self);
  /* Methods introduced in bHYPRE.ProblemDefinition-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    void* self);
  int32_t (*f_Assemble)(
    void* self);
  int32_t (*f_GetObject)(
    void* self,
    struct sidl_BaseInterface__object** A);
  /* Methods introduced in bHYPRE.IJBuildMatrix-v1.0.0 */
  int32_t (*f_SetLocalRange)(
    void* self,
    int32_t ilower,
    int32_t iupper,
    int32_t jlower,
    int32_t jupper);
  int32_t (*f_SetValues)(
    void* self,
    int32_t nrows,
    struct sidl_int__array* ncols,
    struct sidl_int__array* rows,
    struct sidl_int__array* cols,
    struct sidl_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t nrows,
    struct sidl_int__array* ncols,
    struct sidl_int__array* rows,
    struct sidl_int__array* cols,
    struct sidl_double__array* values);
  int32_t (*f_GetLocalRange)(
    void* self,
    int32_t* ilower,
    int32_t* iupper,
    int32_t* jlower,
    int32_t* jupper);
  int32_t (*f_GetRowCounts)(
    void* self,
    int32_t nrows,
    struct sidl_int__array* rows,
    struct sidl_int__array** ncols);
  int32_t (*f_GetValues)(
    void* self,
    int32_t nrows,
    struct sidl_int__array* ncols,
    struct sidl_int__array* rows,
    struct sidl_int__array* cols,
    struct sidl_double__array** values);
  int32_t (*f_SetRowSizes)(
    void* self,
    struct sidl_int__array* sizes);
  int32_t (*f_Print)(
    void* self,
    const char* filename);
  int32_t (*f_Read)(
    void* self,
    const char* filename,
    void* comm);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_IJBuildMatrix__object {
  struct bHYPRE_IJBuildMatrix__epv* d_epv;
  void*                             d_object;
};

#ifdef __cplusplus
}
#endif
#endif
