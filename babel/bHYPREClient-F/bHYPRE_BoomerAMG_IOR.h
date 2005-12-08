/*
 * File:          bHYPRE_BoomerAMG_IOR.h
 * Symbol:        bHYPRE.BoomerAMG-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.BoomerAMG
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_BoomerAMG_IOR_h
#define included_bHYPRE_BoomerAMG_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_bHYPRE_Solver_IOR_h
#include "bHYPRE_Solver_IOR.h"
#endif
#ifndef included_sidl_BaseClass_IOR_h
#include "sidl_BaseClass_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.BoomerAMG" (version 1.0.0)
 * 
 * Algebraic multigrid solver, based on classical Ruge-Stueben.
 * 
 * BoomerAMG requires an IJParCSR matrix
 * 
 * The following optional parameters are available and may be set
 * using the appropriate {\tt Parameter} function (as indicated in
 * parentheses):
 * 
 * \begin{description}
 * 
 * \item[MaxLevels] ({\tt Int}) - maximum number of multigrid
 * levels.
 * 
 * \item[StrongThreshold] ({\tt Double}) - AMG strength threshold.
 * 
 * \item[MaxRowSum] ({\tt Double}) -
 * 
 * \item[CoarsenType] ({\tt Int}) - type of parallel coarsening
 * algorithm used.
 * 
 * \item[MeasureType] ({\tt Int}) - type of measure used; local or
 * global.
 * 
 * \item[CycleType] ({\tt Int}) - type of cycle used; a V-cycle
 * (default) or a W-cycle.
 * 
 * \item[NumGridSweeps] ({\tt IntArray 1D}) - number of sweeps for
 * fine and coarse grid, up and down cycle. DEPRECATED:
 * Use NumSweeps or Cycle?NumSweeps instead.
 * 
 * \item[NumSweeps] ({\tt Int}) - number of sweeps for fine grid, up and
 * down cycle.
 * 
 * \item[Cycle0NumSweeps] ({\tt Int}) - number of sweeps for fine grid
 * 
 * \item[Cycle1NumSweeps] ({\tt Int}) - number of sweeps for down cycle
 * 
 * \item[Cycle2NumSweeps] ({\tt Int}) - number of sweeps for up cycle
 * 
 * \item[Cycle3NumSweeps] ({\tt Int}) - number of sweeps for coarse grid
 * 
 * \item[GridRelaxType] ({\tt IntArray 1D}) - type of smoother used on
 * fine and coarse grid, up and down cycle. DEPRECATED:
 * Use RelaxType or Cycle?RelaxType instead.
 * 
 * \item[RelaxType] ({\tt Int}) - type of smoother for fine grid, up and
 * down cycle.
 * 
 * \item[Cycle0RelaxType] ({\tt Int}) - type of smoother for fine grid
 * 
 * \item[Cycle1RelaxType] ({\tt Int}) - type of smoother for down cycle
 * 
 * \item[Cycle2RelaxType] ({\tt Int}) - type of smoother for up cycle
 * 
 * \item[Cycle3RelaxType] ({\tt Int}) - type of smoother for coarse grid
 * 
 * \item[GridRelaxPoints] ({\tt IntArray 2D}) - point ordering used in
 * relaxation.  DEPRECATED.
 * 
 * \item[RelaxWeight] ({\tt DoubleArray 1D}) - relaxation weight for
 * smoothed Jacobi and hybrid SOR.  DEPRECATED:
 * Instead, use the RelaxWt parameter and the SetLevelRelaxWt function.
 * 
 * \item[RelaxWt] ({\tt Int}) - relaxation weight for all levels for
 * smoothed Jacobi and hybrid SOR.
 * 
 * \item[TruncFactor] ({\tt Double}) - truncation factor for
 * interpolation.
 * 
 * \item[SmoothType] ({\tt Int}) - more complex smoothers.
 * 
 * \item[SmoothNumLevels] ({\tt Int}) - number of levels for more
 * complex smoothers.
 * 
 * \item[SmoothNumSweeps] ({\tt Int}) - number of sweeps for more
 * complex smoothers.
 * 
 * \item[PrintFileName] ({\tt String}) - name of file printed to in
 * association with {\tt SetPrintLevel}.  (not yet implemented).
 * 
 * \item[NumFunctions] ({\tt Int}) - size of the system of PDEs
 * (when using the systems version).
 * 
 * \item[DOFFunc] ({\tt IntArray 1D}) - mapping that assigns the
 * function to each variable (when using the systems version).
 * 
 * \item[Variant] ({\tt Int}) - variant of Schwarz used.
 * 
 * \item[Overlap] ({\tt Int}) - overlap for Schwarz.
 * 
 * \item[DomainType] ({\tt Int}) - type of domain used for Schwarz.
 * 
 * \item[SchwarzRlxWeight] ({\tt Double}) - the smoothing parameter
 * for additive Schwarz.
 * 
 * \item[DebugFlag] ({\tt Int}) -
 * 
 * \end{description}
 * 
 * The following function is specific to this class:
 * 
 * \begin{description}
 * 
 * \item[SetLevelRelxWeight] ({\tt Double , \tt Int}) -
 * relaxation weight for one specified level of smoothed Jacobi and hybrid SOR.
 * 
 * \end{description}
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 */

struct bHYPRE_BoomerAMG__array;
struct bHYPRE_BoomerAMG__object;
struct bHYPRE_BoomerAMG__sepv;

extern struct bHYPRE_BoomerAMG__object*
bHYPRE_BoomerAMG__new(void);

extern struct bHYPRE_BoomerAMG__sepv*
bHYPRE_BoomerAMG__statics(void);

extern void bHYPRE_BoomerAMG__init(
  struct bHYPRE_BoomerAMG__object* self);
extern void bHYPRE_BoomerAMG__fini(
  struct bHYPRE_BoomerAMG__object* self);
extern void bHYPRE_BoomerAMG__IOR_version(int32_t *major, int32_t *minor);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_IJParCSRMatrix__array;
struct bHYPRE_IJParCSRMatrix__object;
struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;

/*
 * Declare the static method entry point vector.
 */

struct bHYPRE_BoomerAMG__sepv {
  /* Implicit builtin methods */
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  /* Methods introduced in bHYPRE.Solver-v1.0.0 */
  /* Methods introduced in bHYPRE.BoomerAMG-v1.0.0 */
  struct bHYPRE_BoomerAMG__object* (*f_Create)(
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* in */ struct bHYPRE_IJParCSRMatrix__object* A);
};

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_BoomerAMG__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE_BoomerAMG__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE_BoomerAMG__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE_BoomerAMG__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE_BoomerAMG__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE_BoomerAMG__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE_BoomerAMG__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE_BoomerAMG__object* self);
  /* Methods introduced in sidl.BaseClass-v0.9.3 */
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_SetIntParameter)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* in */ int32_t value);
  int32_t (*f_SetDoubleParameter)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* in */ double value);
  int32_t (*f_SetStringParameter)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* in */ const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* out */ int32_t* value);
  int32_t (*f_GetDoubleValue)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ const char* name,
    /* out */ double* value);
  int32_t (*f_Setup)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  int32_t (*f_ApplyAdjoint)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE.Solver-v1.0.0 */
  int32_t (*f_SetOperator)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ struct bHYPRE_Operator__object* A);
  int32_t (*f_SetTolerance)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ double tolerance);
  int32_t (*f_SetMaxIterations)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ int32_t max_iterations);
  int32_t (*f_SetLogging)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ int32_t level);
  int32_t (*f_SetPrintLevel)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ int32_t level);
  int32_t (*f_GetNumIterations)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* out */ int32_t* num_iterations);
  int32_t (*f_GetRelResidualNorm)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* out */ double* norm);
  /* Methods introduced in bHYPRE.BoomerAMG-v1.0.0 */
  int32_t (*f_SetLevelRelaxWt)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* in */ double relax_wt,
    /* in */ int32_t level);
  int32_t (*f_InitGridRelaxation)(
    /* in */ struct bHYPRE_BoomerAMG__object* self,
    /* out */ struct sidl_int__array** num_grid_sweeps,
    /* out */ struct sidl_int__array** grid_relax_type,
    /* out */ struct sidl_int__array** grid_relax_points,
    /* in */ int32_t coarsen_type,
    /* out */ struct sidl_double__array** relax_weights,
    /* in */ int32_t max_levels);
};

/*
 * Define the class object structure.
 */

struct bHYPRE_BoomerAMG__object {
  struct sidl_BaseClass__object  d_sidl_baseclass;
  struct bHYPRE_Operator__object d_bhypre_operator;
  struct bHYPRE_Solver__object   d_bhypre_solver;
  struct bHYPRE_BoomerAMG__epv*  d_epv;
  void*                          d_data;
};

struct bHYPRE_BoomerAMG__external {
  struct bHYPRE_BoomerAMG__object*
  (*createObject)(void);

  struct bHYPRE_BoomerAMG__sepv*
  (*getStaticEPV)(void);
  struct sidl_BaseClass__epv*(*getSuperEPV)(void);
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct bHYPRE_BoomerAMG__external*
bHYPRE_BoomerAMG__externals(void);

struct bHYPRE_Solver__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_Solver(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Solver(struct bHYPRE_Solver__object* 
  obj); 

struct bHYPRE_BoomerAMG__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_BoomerAMG(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_BoomerAMG(struct 
  bHYPRE_BoomerAMG__object* obj); 

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_MPICommunicator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj); 

struct bHYPRE_Operator__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_Operator(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Operator(struct 
  bHYPRE_Operator__object* obj); 

struct bHYPRE_IJParCSRMatrix__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_IJParCSRMatrix(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_IJParCSRMatrix(struct 
  bHYPRE_IJParCSRMatrix__object* obj); 

struct sidl_ClassInfo__object* 
  skel_bHYPRE_BoomerAMG_fconnect_sidl_ClassInfo(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj); 

struct bHYPRE_Vector__object* 
  skel_bHYPRE_BoomerAMG_fconnect_bHYPRE_Vector(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_bHYPRE_Vector(struct bHYPRE_Vector__object* 
  obj); 

struct sidl_BaseInterface__object* 
  skel_bHYPRE_BoomerAMG_fconnect_sidl_BaseInterface(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj); 

struct sidl_BaseClass__object* 
  skel_bHYPRE_BoomerAMG_fconnect_sidl_BaseClass(char* url,
  struct sidl_BaseInterface__object **_ex);
char* skel_bHYPRE_BoomerAMG_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj); 

#ifdef __cplusplus
}
#endif
#endif
