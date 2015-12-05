/*
 * File:          bHYPRE_IJBuildMatrix_IOR.h
 * Symbol:        bHYPRE.IJBuildMatrix-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:21 PST
 * Description:   Intermediate Object Representation for bHYPRE.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 85
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJBuildMatrix_IOR_h
#define included_bHYPRE_IJBuildMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
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

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

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
  /* Methods introduced in SIDL.BaseInterface-v0.8.2 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  SIDL_bool (*f_isType)(
    void* self,
    const char* name);
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
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
    struct SIDL_BaseInterface__object** A);
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
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array* values);
  int32_t (*f_AddToValues)(
    void* self,
    int32_t nrows,
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array* values);
  int32_t (*f_GetLocalRange)(
    void* self,
    int32_t* ilower,
    int32_t* iupper,
    int32_t* jlower,
    int32_t* jupper);
  int32_t (*f_GetRowCounts)(
    void* self,
    int32_t nrows,
    struct SIDL_int__array* rows,
    struct SIDL_int__array** ncols);
  int32_t (*f_GetValues)(
    void* self,
    int32_t nrows,
    struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows,
    struct SIDL_int__array* cols,
    struct SIDL_double__array** values);
  int32_t (*f_SetRowSizes)(
    void* self,
    struct SIDL_int__array* sizes);
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
