/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for MLI C interface
 *
 *****************************************************************************/

#ifndef __CINTFACEH__
#define __CINTFACEH__

#include "utilities/utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * new data type definitions
 *****************************************************************************/

typedef struct CMLI_Matrix_Struct CMLI_Matrix;
typedef struct CMLI_Struct        CMLI;
typedef struct CMLI_Vector_Struct CMLI_Vector;
typedef struct CMLI_Solver_Struct CMLI_Solver;
typedef struct CMLI_FEData_Struct CMLI_FEData;
typedef struct CMLI_Method_Struct CMLI_Method;

#include "../util/mli_utils.h"

/******************************************************************************
 * structure prototypes
 *****************************************************************************/

struct CMLI_Struct        { void* mli_; };
struct CMLI_Matrix_Struct { void* matrix_; int owner_; };
struct CMLI_Vector_Struct { void* vector_; int owner_; };
struct CMLI_Solver_Struct { void* solver_; int owner_; };
struct CMLI_FEData_Struct { void* fedata_; int owner_; };
struct CMLI_Method_Struct { void* method_; int owner_; };

/* **************************************************************** */
/* functions for the top level "C" MLI object                       */
/* ---------------------------------------------------------------- */

/* ---------------------------------------------------------------- */
/* constructor and destructor                                       */
/* ---------------------------------------------------------------- */

CMLI *MLI_Create(MPI_Comm comm);

int MLI_Destroy(CMLI *cmli);

/* ---------------------------------------------------------------- */
/* set MLI internal parameters                                      */
/* ---------------------------------------------------------------- */

int MLI_SetTolerance(CMLI *cmli, double tolerance);

int MLI_SetMaxIterations(CMLI *cmli, int iterations);

int MLI_SetNumLevels(CMLI *cmli, int num_levels);

int MLI_SetCyclesAtLevel(CMLI *cmli, int level, int cycles);

/* ---------------------------------------------------------------- */
/* set various operators                                            */
/* ---------------------------------------------------------------- */

int MLI_SetSystemMatrix(CMLI *cmli, int level, CMLI_Matrix *Amat);

int MLI_SetRestriction(CMLI *cmli, int level, CMLI_Matrix *Rmat);

int MLI_SetProlongation(CMLI *cmli, int level, CMLI_Matrix *Pmat);

int MLI_SetFEData(CMLI *cmli, int level, CMLI_FEData *fedata);

int MLI_SetSmoother(CMLI *cmli,int level,int side,CMLI_Solver *solver);

int MLI_SetCoarseSolve(CMLI *cmli, CMLI_Solver *solver);

/* ---------------------------------------------------------------- */
/* set which MG method and the associated parameters                */
/* ---------------------------------------------------------------- */

int MLI_SetMethod(CMLI *cmli, CMLI_Method *method_data);

/* ---------------------------------------------------------------- */
/* set up the multigrid structure (all components)                  */
/* ---------------------------------------------------------------- */

int MLI_Setup(CMLI *cmli);

/* ---------------------------------------------------------------- */
/* solve functions                                                  */
/* ---------------------------------------------------------------- */

int MLI_Cycle(CMLI *cmli, CMLI_Vector *sol, CMLI_Vector *rhs);

int MLI_Solve(CMLI *cmli, CMLI_Vector *sol, CMLI_Vector *rhs);

/* ---------------------------------------------------------------- */
/* diagnostics and statistics                                       */
/* ---------------------------------------------------------------- */

int MLI_SetOutputLevel(CMLI *cmli, int output_level);

int MLI_Print( CMLI *cmli );

int MLI_PrintTiming( CMLI *cmli );

/* **************************************************************** */
/* functions for the "C" finite element data object                 */
/* ---------------------------------------------------------------- */

CMLI_FEData *MLI_FEDataCreate(MPI_Comm comm, void *fedata, char *name);

int MLI_FEDataDestroy(CMLI_FEData *fedata);

/* ---------------------------------------------------------------- */
/* create topological matrices                                      */
/* ---------------------------------------------------------------- */

int MLI_FEDataGetElemNodeHypreMatrix(CMLI_FEData *, MPI_Comm, void **mat);
int MLI_FEDataGetElemFaceHypreMatrix(CMLI_FEData *, MPI_Comm, void **mat);
int MLI_FEDataGetFaceNodeHypreMatrix(CMLI_FEData *, MPI_Comm, void **mat);
int MLI_FEDataGetNodeElemHypreMatrix(CMLI_FEData *, MPI_Comm, void **mat);
int MLI_FEDataGetFaceElemHypreMatrix(CMLI_FEData *, MPI_Comm, void **mat);
int MLI_FEDataGetNodeFaceHypreMatrix(CMLI_FEData *, MPI_Comm, void **mat);

/* **************************************************************** */
/* constructor and destructor for a "C" MLI matrix object           */
/* ---------------------------------------------------------------- */

CMLI_Matrix *MLI_MatrixCreate(void *matrix, char *name, 
                              MLI_Function *destroy_func);

int MLI_MatrixDestroy( CMLI_Matrix *matrix );

/* **************************************************************** */
/* constructor and destructor for a "C" MLI vector object           */
/* ---------------------------------------------------------------- */

CMLI_Vector *MLI_VectorCreate(void *vector, char *name, 
                              MLI_Function *destroy_func);

int MLI_VectorDestroy(CMLI_Vector *vector);

/* **************************************************************** */
/* functions for the "C" MLI solver object                          */
/* ---------------------------------------------------------------- */

CMLI_Solver *MLI_SolverCreate(char *name);

int MLI_SolverDestroy(CMLI_Solver *solver);

int MLI_SolverSetParams(CMLI_Solver *solver, char *param_string,
                        int argc, char **argv);

/* **************************************************************** */
/* functions for the "C" MLI Method object                          */
/* ---------------------------------------------------------------- */

CMLI_Method *MLI_MethodCreate(char *name, MPI_Comm comm);

int MLI_MethodDestroy(CMLI_Method *method);

int MLI_MethodSetParams(CMLI_Method *cmethod, char *name, 
                        int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif

