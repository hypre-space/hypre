/*
 * File:          Hypre_IJBuildMatrix_IOR.h
 * Symbol:        Hypre.IJBuildMatrix-v0.1.6
 * Symbol Type:   interface
 * Babel Version: 0.8.0
 * SIDL Created:  20030121 14:39:22 PST
 * Generated:     20030121 14:39:26 PST
 * Description:   Intermediate Object Representation for Hypre.IJBuildMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 155
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_IJBuildMatrix_IOR_h
#define included_Hypre_IJBuildMatrix_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.IJBuildMatrix" (version 0.1.6)
 * 
 * 
 * This interface represents a linear-algebraic conceptual view of a
 * linear system.  The 'I' and 'J' in the name are meant to be
 * mnemonic for the traditional matrix notation A(I,J).
 * 
 * 
 */

struct Hypre_IJBuildMatrix__array;
struct Hypre_IJBuildMatrix__object;

extern struct Hypre_IJBuildMatrix__object*
Hypre_IJBuildMatrix__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_IJBuildMatrix__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
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
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.6 */
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
  /* Methods introduced in Hypre.IJBuildMatrix-v0.1.6 */
  int32_t (*f_Create)(
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
  int32_t (*f_SetRowSizes)(
    void* self,
    struct SIDL_int__array* sizes);
  int32_t (*f_SetDiagOffdSizes)(
    void* self,
    struct SIDL_int__array* diag_sizes,
    struct SIDL_int__array* offdiag_sizes);
  int32_t (*f_Read)(
    void* self,
    const char* filename,
    void* comm);
  int32_t (*f_Print)(
    void* self,
    const char* filename);
};

/*
 * Define the interface object structure.
 */

struct Hypre_IJBuildMatrix__object {
  struct Hypre_IJBuildMatrix__epv* d_epv;
  void*                            d_object;
};

#ifdef __cplusplus
}
#endif
#endif
