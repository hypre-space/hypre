/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.19 $
 ***********************************************************************EHEADER*/




#include "./HYPRE_parcsr_ls.h"
#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"
#include "../utilities/mpistubs.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"

/* AB 8/06 - replace header file */
/* #include "../parcsr_mv/par_vector.h" */
#include "../parcsr_mv/_hypre_parcsr_mv.h"


  /* These are what we need from Euclid */
#include "../distributed_ls/Euclid/Mem_dh.h"
#include "../distributed_ls/Euclid/io_dh.h"
#include "../distributed_ls/Euclid/TimeLog_dh.h"
#include "../distributed_ls/Euclid/Parser_dh.h"
#include "../distributed_ls/Euclid/Euclid_dh.h"

/*------------------------------------------------------------------
 * Error checking
 *------------------------------------------------------------------*/

#define HYPRE_EUCLID_ERRCHKA \
          if (errFlag_dh) {  \
            setError_dh("", __FUNC__, __FILE__, __LINE__); \
            printErrorMsg(stderr);  \
            hypre_MPI_Abort(comm_dh, -1); \
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
HYPRE_Int 
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
  hypre_MPI_Comm_size(comm_dh, &np_dh);    HYPRE_EUCLID_ERRCHKA;
  hypre_MPI_Comm_rank(comm_dh, &myid_dh);  HYPRE_EUCLID_ERRCHKA;

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
HYPRE_Int 
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
  /*if (parser_dh != NULL) { dah 3/16/06  */
  if (parser_dh != NULL && ref_counter == 0) {
    Parser_dhDestroy(parser_dh); HYPRE_EUCLID_ERRCHKA;
    parser_dh = NULL;
  }

  /*if (tlog_dh != NULL) {  dah 3/16/06  */
  if (tlog_dh != NULL && ref_counter == 0) {
    TimeLog_dhDestroy(tlog_dh); HYPRE_EUCLID_ERRCHKA;
    tlog_dh = NULL;
  }

  /*------------------------------------------------------------------ 
     optionally print Euclid's memory report, 
     then destroy the memory object.
   *------------------------------------------------------------------ */
  /*if (mem_dh != NULL) {  dah 3/16/06  */
  if (mem_dh != NULL && ref_counter == 0) {
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
HYPRE_Int 
HYPRE_EuclidSetup( HYPRE_Solver solver,
                         HYPRE_ParCSRMatrix A,
                         HYPRE_ParVector b,
                         HYPRE_ParVector x   )
{
  START_FUNC_DH
  Euclid_dh eu = (Euclid_dh)solver;


#if 0

for testing!
  {
  HYPRE_Int ierr;
  HYPRE_Int m,n,rs,re,cs,ce;

   HYPRE_DistributedMatrix mat;
   ierr = HYPRE_ConvertParCSRMatrixToDistributedMatrix( A, &mat );
   if (ierr) exit(-1);

    ierr = HYPRE_DistributedMatrixGetDims(mat, &m, &n);
    ierr = HYPRE_DistributedMatrixGetLocalRange(mat, &rs, &re,
                                       &cs, &ce);

    hypre_printf("\n### [%i] m= %i, n= %i, rs= %i, re= %i, cs= %i, ce= %i\n",
                                            myid_dh, m, n, rs, re, cs, ce);

   ierr = HYPRE_DistributedMatrixDestroy(mat);

   if (ierr) exit(-1);
  }
#endif

  Euclid_dhInputHypreMat(eu, A); HYPRE_EUCLID_ERRCHKA;
  Euclid_dhSetup(eu); HYPRE_EUCLID_ERRCHKA;

  END_FUNC_VAL(0)
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

#undef __FUNC__
#define __FUNC__ "HYPRE_EuclidSolve"
HYPRE_Int 
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
HYPRE_Int
HYPRE_EuclidSetParams(HYPRE_Solver solver, 
                            HYPRE_Int argc,
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
HYPRE_Int
HYPRE_EuclidSetParamsFromFile(HYPRE_Solver solver, 
                                    char *filename )
{
  START_FUNC_DH
  Parser_dhUpdateFromFile(parser_dh, filename); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

HYPRE_Int
HYPRE_EuclidSetLevel(HYPRE_Solver solver, 
				HYPRE_Int level)
{
  char str_level[8];
  START_FUNC_DH
  hypre_sprintf(str_level,"%d",level);
  Parser_dhInsert(parser_dh, "-level", str_level); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

HYPRE_Int
HYPRE_EuclidSetBJ(HYPRE_Solver solver, 
				HYPRE_Int bj)
{
  char str_bj[8];
  START_FUNC_DH
  hypre_sprintf(str_bj,"%d",bj);
  Parser_dhInsert(parser_dh, "-bj", str_bj); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

HYPRE_Int
HYPRE_EuclidSetStats(HYPRE_Solver solver, 
				HYPRE_Int eu_stats)
{
  char str_eu_stats[8];
  START_FUNC_DH
  hypre_sprintf(str_eu_stats,"%d",eu_stats);
  Parser_dhInsert(parser_dh, "-eu_stats", str_eu_stats); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

HYPRE_Int
HYPRE_EuclidSetMem(HYPRE_Solver solver, 
				HYPRE_Int eu_mem)
{
  char str_eu_mem[8];
  START_FUNC_DH
  hypre_sprintf(str_eu_mem,"%d",eu_mem);
  Parser_dhInsert(parser_dh, "-eu_mem", str_eu_mem); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

HYPRE_Int
HYPRE_EuclidSetILUT(HYPRE_Solver solver, 
				double ilut)
{
  char str_ilut[256];
  START_FUNC_DH
  hypre_sprintf(str_ilut,"%f",ilut);
  Parser_dhInsert(parser_dh, "-ilut", str_ilut); HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

HYPRE_Int
HYPRE_EuclidSetSparseA(HYPRE_Solver solver, 
				double sparse_A)
{
  char str_sparse_A[256];
  START_FUNC_DH
  hypre_sprintf(str_sparse_A,"%f",sparse_A);
  Parser_dhInsert(parser_dh, "-sparseA", str_sparse_A); 
  HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}

HYPRE_Int
HYPRE_EuclidSetRowScale(HYPRE_Solver solver, 
				HYPRE_Int row_scale)
{
  char str_row_scale[8];
  START_FUNC_DH
  hypre_sprintf(str_row_scale,"%d",row_scale);
  Parser_dhInsert(parser_dh, "-rowScale", str_row_scale); 
  HYPRE_EUCLID_ERRCHKA;
  END_FUNC_VAL(0)
}
