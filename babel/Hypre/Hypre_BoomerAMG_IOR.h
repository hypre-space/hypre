/*
 * File:          Hypre_BoomerAMG_IOR.h
 * Symbol:        Hypre.BoomerAMG-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:13 PST
 * Description:   Intermediate Object Representation for Hypre.BoomerAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.0
 * source-line   = 1232
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_BoomerAMG_IOR_h
#define included_Hypre_BoomerAMG_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Operator_IOR_h
#include "Hypre_Operator_IOR.h"
#endif
#ifndef included_Hypre_Solver_IOR_h
#include "Hypre_Solver_IOR.h"
#endif
#ifndef included_SIDL_BaseClass_IOR_h
#include "SIDL_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.BoomerAMG" (version 0.1.7)
 * 
 * Algebraic multigrid solver, based on classical Ruge-Stueben.
 * 
 * The following optional parameters are available and may be set
 * using the appropriate {\tt Parameter} function (as indicated in
 * parentheses):
 * 
 * \begin{description}
 * 
 * \item[Max Levels] ({\tt Int}) - maximum number of multigrid
 * levels.
 * 
 * \item[Strong Threshold] ({\tt Double}) - AMG strength threshold.
 * 
 * \item[Max Row Sum] ({\tt Double}) -
 * 
 * \item[Coarsen Type] ({\tt Int}) - type of parallel coarsening
 * algorithm used.
 * 
 * \item[Measure Type] ({\tt Int}) - type of measure used; local or
 * global.
 * 
 * \item[Cycle Type] ({\tt Int}) - type of cycle used; a V-cycle
 * (default) or a W-cycle.
 * 
 * \item[Num Grid Sweeps] ({\tt IntArray}) - number of sweeps for
 * fine and coarse grid, up and down cycle.
 * 
 * \item[Grid Relax Type] ({\tt IntArray}) - type of smoother used
 * on fine and coarse grid, up and down cycle.
 * 
 * \item[Grid Relax Points] ({\tt IntArray}) - point ordering used
 * in relaxation.
 * 
 * \item[Relax Weight] ({\tt DoubleArray}) - relaxation weight for
 * smoothed Jacobi and hybrid SOR.
 * 
 * \item[Truncation Factor] ({\tt Double}) - truncation factor for
 * interpolation.
 * 
 * \item[Smooth Type] ({\tt Int}) - more complex smoothers.
 * 
 * \item[Smooth Num Levels] ({\tt Int}) - number of levels for more
 * complex smoothers.
 * 
 * \item[Smooth Num Sweeps] ({\tt Int}) - number of sweeps for more
 * complex smoothers.
 * 
 * \item[Print File Name] ({\tt String}) - name of file printed to
 * in association with {\tt SetPrintLevel}.  (not yet implemented).
 * 
 * \item[Num Functions] ({\tt Int}) - size of the system of PDEs
 * (when using the systems version).
 * 
 * \item[DOF Func] ({\tt IntArray}) - mapping that assigns the
 * function to each variable (when using the systems version).
 * 
 * \item[Variant] ({\tt Int}) - variant of Schwarz used.
 * 
 * \item[Overlap] ({\tt Int}) - overlap for Schwarz.
 * 
 * \item[Domain Type] ({\tt Int}) - type of domain used for
 * Schwarz.
 * 
 * \item[Schwarz Relaxation Weight] ({\tt Double}) - the smoothing
 * parameter for additive Schwarz.
 * 
 * \item[Debug Flag] ({\tt Int}) -
 * 
 * \end{description}
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Changed name from 'ParAMG' (x)
 * 
 */

struct Hypre_BoomerAMG__array;
struct Hypre_BoomerAMG__object;

extern struct Hypre_BoomerAMG__object*
Hypre_BoomerAMG__new(void);

extern struct Hypre_BoomerAMG__object*
Hypre_BoomerAMG__remote(const char *url);

extern void Hypre_BoomerAMG__init(
  struct Hypre_BoomerAMG__object* self);
extern void Hypre_BoomerAMG__fini(
  struct Hypre_BoomerAMG__object* self);
extern void Hypre_BoomerAMG__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;
struct SIDL_ClassInfo__array;
struct SIDL_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_BoomerAMG__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    struct Hypre_BoomerAMG__object* self,
    const char* name);
  void (*f__delete)(
    struct Hypre_BoomerAMG__object* self);
  void (*f__ctor)(
    struct Hypre_BoomerAMG__object* self);
  void (*f__dtor)(
    struct Hypre_BoomerAMG__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  void (*f_addRef)(
    struct Hypre_BoomerAMG__object* self);
  void (*f_deleteRef)(
    struct Hypre_BoomerAMG__object* self);
  SIDL_bool (*f_isSame)(
    struct Hypre_BoomerAMG__object* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInt)(
    struct Hypre_BoomerAMG__object* self,
    const char* name);
  SIDL_bool (*f_isType)(
    struct Hypre_BoomerAMG__object* self,
    const char* name);
  /* Methods introduced in SIDL.BaseClass-v0.8.1 */
  struct SIDL_ClassInfo__object* (*f_getClassInfo)(
    struct Hypre_BoomerAMG__object* self);
  /* Methods introduced in SIDL.BaseInterface-v0.8.1 */
  /* Methods introduced in Hypre.Operator-v0.1.7 */
  int32_t (*f_SetCommunicator)(
    struct Hypre_BoomerAMG__object* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    struct Hypre_BoomerAMG__object* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    struct Hypre_BoomerAMG__object* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    struct Hypre_BoomerAMG__object* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    struct Hypre_BoomerAMG__object* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    struct Hypre_BoomerAMG__object* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_GetIntValue)(
    struct Hypre_BoomerAMG__object* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    struct Hypre_BoomerAMG__object* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    struct Hypre_BoomerAMG__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object* x);
  int32_t (*f_Apply)(
    struct Hypre_BoomerAMG__object* self,
    struct Hypre_Vector__object* b,
    struct Hypre_Vector__object** x);
  /* Methods introduced in Hypre.Solver-v0.1.7 */
  int32_t (*f_SetOperator)(
    struct Hypre_BoomerAMG__object* self,
    struct Hypre_Operator__object* A);
  int32_t (*f_SetTolerance)(
    struct Hypre_BoomerAMG__object* self,
    double tolerance);
  int32_t (*f_SetMaxIterations)(
    struct Hypre_BoomerAMG__object* self,
    int32_t max_iterations);
  int32_t (*f_SetLogging)(
    struct Hypre_BoomerAMG__object* self,
    int32_t level);
  int32_t (*f_SetPrintLevel)(
    struct Hypre_BoomerAMG__object* self,
    int32_t level);
  int32_t (*f_GetNumIterations)(
    struct Hypre_BoomerAMG__object* self,
    int32_t* num_iterations);
  int32_t (*f_GetRelResidualNorm)(
    struct Hypre_BoomerAMG__object* self,
    double* norm);
  /* Methods introduced in Hypre.BoomerAMG-v0.1.7 */
};

/*
 * Define the class object structure.
 */

struct Hypre_BoomerAMG__object {
  struct SIDL_BaseClass__object d_sidl_baseclass;
  struct Hypre_Operator__object d_hypre_operator;
  struct Hypre_Solver__object   d_hypre_solver;
  struct Hypre_BoomerAMG__epv*  d_epv;
  void*                         d_data;
};

struct Hypre_BoomerAMG__external {
  struct Hypre_BoomerAMG__object*
  (*createObject)(void);

  struct Hypre_BoomerAMG__object*
  (*createRemote)(const char *url);

};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 * loading DLLs
 */

const struct Hypre_BoomerAMG__external*
Hypre_BoomerAMG__externals(void);

#ifdef __cplusplus
}
#endif
#endif
