#include "./HYPRE_parcsr_ls.h"
#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"
#include "../utilities/mpistubs.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"
#include "../parcsr_mv/par_vector.h"


  /* These are what we need from Euclid */
#include "../distributed_ls/Euclid/Euclid_dh.h"
#include "../distributed_ls/Euclid/Mem_dh.h"
#include "../distributed_ls/Euclid/io_dh.h"

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
   *       src/distributed_ls/Euclid/macros_dh.h and
   *       src/distributed_ls/Euclid/src/globalObjects.c
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
 * HYPRE_EuclidCreate - Return a Euclid "solver".  
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidCreate"
int 
HYPRE_EuclidCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
  START_FUNC_DH
  Euclid_dh eu; 

  /*----------------------------------------------------------- 
   * create a few global objects (yuck!) for Euclid's use;
   * these  are all pointers, are initially NULL, and are be set 
   * back to NULL in HYPRE_EuclidDestroy()
   * Global objects are defined in 
   * src/distributed_ls/Euclid/src/globalObjects.c
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
 * HYPRE_EuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidDestroy"
int 
HYPRE_EuclidDestroy( HYPRE_Solver solver )
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  bool printMemReport = false;
  bool printStats = false;
  bool logging = eu->logging;

  /*---------------------------------------------------------------- 
     this block is for printing test data; this is used
     for diffing in autotests.
   *---------------------------------------------------------------- */
  if (Parser_dhHasSwitch(parser_dh, "-printTestData")) {
    FILE *fp;

    /* get filename to which to write report */
    char fname[] = "test_data_dh.temp", *fnamePtr = fname;
    Parser_dhReadString(parser_dh, "-printTestData", &fnamePtr); HYPRE_EUCLID_ERRCHKA;
    if (!strcmp(fnamePtr, "1")) {  /* in case usr didn't supply a name! */
      fnamePtr = fname;
    }

    /* print the report */
    fp = openFile_dh(fnamePtr, "w"); HYPRE_EUCLID_ERRCHKA;
    Euclid_dhPrintTestData(eu, fp); HYPRE_EUCLID_ERRCHKA;
    closeFile_dh(fp); HYPRE_EUCLID_ERRCHKA;
   
    printf_dh("\n@@@@@ Euclid test data was printed to file: %s\n\n", fnamePtr);
  }


  /*---------------------------------------------------------------- 
     determine which of Euclid's internal reports to print
   *----------------------------------------------------------------*/
  if (logging) {
    printStats = true;
    printMemReport = true;
  }
  if (parser_dh != NULL) {
    if (Parser_dhHasSwitch(parser_dh, "-eu_stats")) {
      printStats = true;
    }
    if (Parser_dhHasSwitch(parser_dh, "-eu_mem")) {
      printMemReport = true;
    }
  }

  /*------------------------------------------------------------------ 
     print Euclid's internal report, then destroy the Euclid object 
   *------------------------------------------------------------------ */
  if (printStats) {
    Euclid_dhPrintHypreReport(eu, stdout); HYPRE_EUCLID_ERRCHKA;
  }
  Euclid_dhDestroy(eu); HYPRE_EUCLID_ERRCHKA;


  /*------------------------------------------------------------------ 
     destroy all remaining Euclid library objects 
     (except the memory object)
   *------------------------------------------------------------------ */
  if (parser_dh != NULL) {
    Parser_dhDestroy(parser_dh); HYPRE_EUCLID_ERRCHKA;
    parser_dh = NULL;
  }

  if (tlog_dh != NULL) {
    TimeLog_dhDestroy(tlog_dh); HYPRE_EUCLID_ERRCHKA;
    tlog_dh = NULL;
  }

  /*------------------------------------------------------------------ 
     optionally print Euclid's memory report, 
     then destroy the memory object.
   *------------------------------------------------------------------ */
  if (mem_dh != NULL) {
    if (printMemReport) { 
      Mem_dhPrint(mem_dh, stdout, false); HYPRE_EUCLID_ERRCHKA; 
    }
    Mem_dhDestroy(mem_dh);  HYPRE_EUCLID_ERRCHKA;
    mem_dh = NULL;
  }

  #ifdef ENABLE_EUCLID_LOGGING
  closeLogfile_dh(); HYPRE_EUCLID_ERRCHKA;
  #endif

  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidSetup"
int 
HYPRE_EuclidSetup( HYPRE_Solver solver,
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
 * HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidSolve"
int 
HYPRE_EuclidSolve( HYPRE_Solver solver,
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
#define __FUNC__ "HYPRE_EuclidSetParams"
int
HYPRE_EuclidSetParams(HYPRE_Solver solver, 
                            int argc,
                            char *argv[] )
{
  START_FUNC_DH
  Parser_dhInit(parser_dh, argc, argv); HYPRE_EUCLID_ERRCHKA;

  /* maintainers note: even though Parser_dhInit() was called in
     HYPRE_EuclidCreate(), it's O.K. to call it again.
   */
  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * Insert (flag, value) pairs in Euclid's  database from file
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidSetParamsFromFile"
int
HYPRE_EuclidSetParamsFromFile(HYPRE_Solver solver, 
                                    char *filename )
{
  START_FUNC_DH
  Parser_dhUpdateFromFile(parser_dh, filename); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidPrintParams - prints summary of current settings and
 * other info.  Call this after HYPRE_EuclidSetup() completes
 * (you can call it before, but info won't be accurate).
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidPrintParams"
int
HYPRE_EuclidPrintParams(HYPRE_Solver solver)
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  Euclid_dhPrintHypreReport(eu, stdout); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}


#if 0

/*--------------------------------------------------------------------------
 * HYPRE_EuclidReadRho
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidReadRho"
int
HYPRE_EuclidReadRho(HYPRE_Solver solver, double *rho)
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  *rho = eu->rho_final;
  END_FUNC_VAL(0)
}


/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetLogging 
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidSetLogging"
int
HYPRE_EuclidSetLogging(HYPRE_Solver solver, 
                             int logging)
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;
  eu->logging = logging;
  END_FUNC_VAL(0)
}

#endif
