#include "./HYPRE_parcsr_ls.h"
#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"
#include "../utilities/mpistubs.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_matrix_vector/vector.h"
#include "../parcsr_matrix_vector/par_vector.h"


  /* These are what we need from Euclid */
#include "../distributed_linear_solvers/Euclid/include/Euclid_dh.h"
#include "../distributed_linear_solvers/Euclid/include/Mem_dh.h"

/*------------------------------------------------------------------
 * Error checking
 *------------------------------------------------------------------*/

#define HYPRE_EUCLID_ERRCHKA \
          if (errFlag_dh) {  \
            setError_dh("", __FUNC__, __FILE__, __LINE__); \
            printErrorMsg(stderr);  \
            MPI_Abort(comm_dh, -1); \
          }

  /* What is best to do here?  
   * What is HYPRE's error checking strategy?  
   * The shadow knows . . .
   *
   * Note: HYPRE_EUCLID_ERRCHKA macro is only used within this file.
   *
   * Note: "printErrorMsg(stderr)" is O.K. for debugging and
   *        development, possibly not for production.  This
   *        call causes Euclid to print a function call stack
   *        trace that led to the error.  (Potentially, each
   *        MPI task could print a trace.)
   *
   * Note: the __FUNC__ defines at the beginning of the function
   *       calls are used in Euclid's internal error-checking scheme.
   *       The "START_FUNC_DH" and "END_FUNC_VAL" macros are
   *       used for debugging: when "logFuncsToStderr == true"
   *       a function call trace is force-written to stderr;
   *       (useful for debugging over dial-up lines!)  See
   *       src/distributed_linear_solvers/Euclid/macros_dh.h and
   *       src/distributed_linear_solvers/Euclid/src/globalObjects.c
   *       for further info.
   */


/*--------------------------------------------------------------------------
 * debugging: if ENABLE_EUCLID_LOGGING is defined, each MPI task will open 
 * "logFile.id" for writing; also, function-call tracing is operational
 * (ie, you can set logFuncsToFile = true, logFuncsToSterr = true).
 *
 *--------------------------------------------------------------------------*/
#undef ENABLE_EUCLID_LOGGING

#if !defined(ENABLE_EUCLID_LOGGING)
#undef START_FUNC_DH
#undef END_FUNC_VAL
#undef END_FUNC_DH
#define START_FUNC_DH     /**/
#define END_FUNC_DH       /**/
#define END_FUNC_VAL(a)   return(a);
#endif


/*--------------------------------------------------------------------------
 * HYPRE_ParCSREuclidCreate - Return a Euclid "solver".  
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidCreate"
int 
HYPRE_ParCSREuclidCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
  START_FUNC_DH
  Euclid_dh eu; 

  /*----------------------------------------------------------- 
   * create a few global objects (yuck!) for Euclid's use;
   * these  are all pointers, are initially NULL, and are be set 
   * back to NULL in HYPRE_ParCSREuclidDestroy()
   * Global objects are defined in 
   * src/distributed_linear_solvers/Euclid/src/globalObjects.c
   *-----------------------------------------------------------*/

  comm_dh = comm;
  MPI_Comm_size(comm_dh, &np_dh);    HYPRE_EUCLID_ERRCHKA;
  MPI_Comm_rank(comm_dh, &myid_dh);  HYPRE_EUCLID_ERRCHKA;

  #ifdef ENABLE_EUCLID_LOGGING
  openLogfile_dh(0, NULL); HYPRE_EUCLID_ERRCHKA;
  #endif

  if (mem_dh == NULL) {
    Mem_dhCreate(&mem_dh);  HYPRE_EUCLID_ERRCHKA;
  }

  if (tlog_dh == NULL) {
    TimeLog_dhCreate(&tlog_dh); HYPRE_EUCLID_ERRCHKA;
  }

  if (parser_dh == NULL) {
    Parser_dhCreate(&parser_dh); HYPRE_EUCLID_ERRCHKA;
  }
  Parser_dhInit(parser_dh, 0, NULL); HYPRE_EUCLID_ERRCHKA;

  /*----------------------------------------------------------- 
   * create and return a Euclid object
   *-----------------------------------------------------------*/
  Euclid_dhCreate(&eu); HYPRE_EUCLID_ERRCHKA;
  *solver = (HYPRE_Solver) eu;

  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSREuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidDestroy"
int 
HYPRE_ParCSREuclidDestroy( HYPRE_Solver solver )
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  bool printMemReport;

  Euclid_dhDestroy(eu); HYPRE_EUCLID_ERRCHKA;

  if (parser_dh != NULL) {
    printMemReport = Parser_dhHasSwitch(parser_dh, "-printMemReport"); HYPRE_EUCLID_ERRCHKA;
    Parser_dhDestroy(parser_dh); HYPRE_EUCLID_ERRCHKA;
    parser_dh = NULL;
  }

  if (tlog_dh != NULL) {
    TimeLog_dhDestroy(tlog_dh); HYPRE_EUCLID_ERRCHKA;
    tlog_dh = NULL;
  }

  if (mem_dh != NULL) {
    if (printMemReport) { Mem_dhPrint(mem_dh, stdout, false); }
    Mem_dhDestroy(mem_dh);  HYPRE_EUCLID_ERRCHKA;
    mem_dh = NULL;
  }

  #ifdef ENABLE_EUCLID_LOGGING
  closeLogfile_dh(); HYPRE_EUCLID_ERRCHKA;
  #endif

  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSREuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidSetup"
int 
HYPRE_ParCSREuclidSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x   )
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;

  Euclid_dhInputHypreMat(eu, A); HYPRE_EUCLID_ERRCHKA;
  Euclid_dhSetup(eu); HYPRE_EUCLID_ERRCHKA;

  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSREuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidSolve"
int 
HYPRE_ParCSREuclidSolve( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector bb,
                         HYPRE_ParVector xx  )
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  double *b, *x;

  x = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) bb));
  b = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) xx));

  Euclid_dhApply(eu, x, b); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * Insert command line (flag, value) pairs in Euclid's 
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidSetParams"
int
HYPRE_ParCSREuclidSetParams(HYPRE_Solver solver, 
                            int argc,
                            char *argv[] )
{
  START_FUNC_DH
  Parser_dhInit(parser_dh, argc, argv); HYPRE_EUCLID_ERRCHKA;

  /* maintainers note: even though Parser_dhInit() was called in
     HYPRE_ParCSREuclidCreate(), it's O.K. to call it again.
   */
  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * Insert (flag, value) pairs in Euclid's  database from file
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidSetParamsFromFile"
int
HYPRE_ParCSREuclidSetParamsFromFile(HYPRE_Solver solver, 
                                    char *filename )
{
  START_FUNC_DH
  Parser_dhUpdateFromFile(parser_dh, filename); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSREuclidPrintParams - prints summary of current settings and
 * other info.  Call this after HYPRE_ParCSREuclidSetup() completes
 * (you can call it before, but info won't be accurate).
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidPrintParams"
int
HYPRE_ParCSREuclidPrintParams(HYPRE_Solver solver)
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  Euclid_dhPrintParams(eu, stdout); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSREuclidSetLogging 
 *--------------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidReadRho"
int
HYPRE_ParCSREuclidReadRho(HYPRE_Solver solver, double *rho)
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  *rho = eu->rho_final;
  END_FUNC_VAL(0)
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSREuclidSetLogging 
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_ParCSREuclidSetLogging"
int
HYPRE_ParCSREuclidSetLogging(HYPRE_Solver solver, 
                             int logging)
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  eu->logging = logging;
  END_FUNC_VAL(0)
}
