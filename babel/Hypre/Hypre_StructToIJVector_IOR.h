/*
 * File:          Hypre_StructToIJVector_IOR.h
 * Symbol:        Hypre.StructToIJVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:30 PST
 * Description:   Intermediate Object Representation for Hypre.StructToIJVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 450
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_StructToIJVector_IOR_h
#define included_Hypre_StructToIJVector_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_ProblemDefinition_IOR_h
#include "Hypre_ProblemDefinition_IOR.h"
#endif
#ifndef included_Hypre_StructuredGridBuildVector_IOR_h
#include "Hypre_StructuredGridBuildVector_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.StructToIJVector" (version 0.1.5)
 */

struct Hypre_StructToIJVector__array;
struct Hypre_StructToIJVector__object;

extern struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__new(void);

extern struct Hypre_StructToIJVector__object*
Hypre_StructToIJVector__remote(const char *url);

extern void Hypre_StructToIJVector__init(
  struct Hypre_StructToIJVector__object* self);
extern void Hypre_StructToIJVector__fini(
  struct Hypre_StructToIJVector__object* self);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_IJBuildVector__array;
struct Hypre_IJBuildVector__object;
struct Hypre_StructGrid__array;
struct Hypre_StructGrid__object;
struct Hypre_StructStencil__array;
struct Hypre_StructStencil__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_StructToIJVector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_StructToIJVector__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_StructToIJVector__object* self);
  void (*f__ctor)(
    struct Hypre_StructToIJVector__object* self);
  void (*f__dtor)(
    struct Hypre_StructToIJVector__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.7.4 */
  void (*f_addReference)(
    struct Hypre_StructToIJVector__object* self);
  void (*f_deleteReference)(
    struct Hypre_StructToIJVector__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    struct Hypre_StructToIJVector__object* self,
    const char* name);
  SIDL_bool (*f_isInstanceOf)(
    struct Hypre_StructToIJVector__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.7.4 */
  /* Methods introduced in SIDL.BaseInterface-v0.7.4 */
  /* Methods introduced in Hypre.ProblemDefinition-v0.1.5 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_StructToIJVector__object* self,
    void* mpi_comm);
  int32_t (*f_Initialize)(
    struct Hypre_StructToIJVector__object* self);
  int32_t (*f_Assemble)(
    struct Hypre_StructToIJVector__object* self);
  int32_t (*f_GetObject)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_BaseInterface__object** A);
  /* Methods introduced in Hypre.StructuredGridBuildVector-v0.1.5 */
  int32_t (*f_SetGrid)(
    struct Hypre_StructToIJVector__object* self,
    struct Hypre_StructGrid__object* grid);
  int32_t (*f_SetStencil)(
    struct Hypre_StructToIJVector__object* self,
    struct Hypre_StructStencil__object* stencil);
  int32_t (*f_SetValue)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_int__array* grid_index,
    double value);
  int32_t (*f_SetBoxValues)(
    struct Hypre_StructToIJVector__object* self,
    struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper,
    struct SIDL_double__array* values);
  /* Methods introduced in Hypre.StructToIJVector-v0.1.5 */
  int32_t (*f_SetIJVector)(
    struct Hypre_StructToIJVector__object* self,
    struct Hypre_IJBuildVector__object* I);
};

/*
 * Define the class object structure.
 */

struct Hypre_StructToIJVector__object {
  struct SIDL_BaseClass__object                  d_sidl_baseclass;
  struct Hypre_ProblemDefinition__object         d_hypre_problemdefinition;
  struct Hypre_StructuredGridBuildVector__object 
    d_hypre_structuredgridbuildvector;
  struct Hypre_StructToIJVector__epv*            d_epv;
  void*                                          d_data;
};

struct Hypre_StructToIJVector__external {
  struct Hypre_StructToIJVector__object*
  (*createObject)(void);

  struct Hypre_StructToIJVector__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_StructToIJVector__external*
Hypre_StructToIJVector__externals(void);

#ifdef __cplusplus
}
#endif
#endif
