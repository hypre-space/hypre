/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __HYPRE_FEI__
#define __HYPRE_FEI__

#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"

#ifdef __cplusplus
extern "C"
{
#endif

#include "HYPRE_MHMatrix.h"

typedef struct HYPRE_LSI_DDIlut_Struct
{
   MPI_Comm  comm;
   MH_Matrix *mh_mat;
   double    thresh;
   double    fillin;
   int       overlap;
   int       Nrows;
   int       extNrows;
   int       *mat_ia;
   int       *mat_ja;
   double    *mat_aa;
   int       outputLevel;
   int       reorder;
   int       *order_array;
   int       *reorder_array;
}
HYPRE_LSI_DDIlut;

/* HYPRE_LSI_ddict.c */
int HYPRE_LSI_DDICTCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_LSI_DDICTDestroy( HYPRE_Solver solver );
int HYPRE_LSI_DDICTSetFillin( HYPRE_Solver solver, double fillin);
int HYPRE_LSI_DDICTSetOutputLevel( HYPRE_Solver solver, int level);
int HYPRE_LSI_DDICTSetDropTolerance( HYPRE_Solver solver, double thresh);
int HYPRE_LSI_DDICTSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
						  HYPRE_ParVector b,   HYPRE_ParVector x );
int HYPRE_LSI_DDICTSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
						  HYPRE_ParVector b,   HYPRE_ParVector x );
/* HYPRE_LSI_ddilut.c */
int HYPRE_LSI_DDIlutCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_LSI_DDIlutDestroy( HYPRE_Solver solver );
int HYPRE_LSI_DDIlutSetFillin( HYPRE_Solver solver, double fillin);
int HYPRE_LSI_DDIlutSetDropTolerance( HYPRE_Solver solver, double thresh);
int HYPRE_LSI_DDIlutSetOverlap( HYPRE_Solver solver );
int HYPRE_LSI_DDIlutSetReorder( HYPRE_Solver solver );
int HYPRE_LSI_DDIlutSetOutputLevel( HYPRE_Solver solver, int level);
int HYPRE_LSI_DDIlutSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
						   HYPRE_ParVector b,   HYPRE_ParVector x );
int HYPRE_LSI_DDIlutSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
						   HYPRE_ParVector b,   HYPRE_ParVector x );
int HYPRE_LSI_DDIlutGetRowLengths(MH_Matrix *Amat, int *leng, int **recv_leng,
                                  MPI_Comm mpi_comm);
int HYPRE_LSI_DDIlutGetOffProcRows(MH_Matrix *Amat, int leng, int *recv_leng,
                           int Noffset, int *map, int *map2, int **int_buf,
								   double **dble_buf, MPI_Comm mpi_comm);
int HYPRE_LSI_DDIlutComposeOverlappedMatrix(MH_Matrix *mh_mat, 
											int *total_recv_leng, int **recv_lengths, int **int_buf, 
											double **dble_buf, int **sindex_array, int **sindex_array2, 
											int *offset, MPI_Comm mpi_comm);
int HYPRE_LSI_DDIlutDecompose(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
							  int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa, 
							  int *map, int *map2, int Noffset);
int HYPRE_LSI_DDIlutDecompose2(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
							   int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa, 
							   int *map, int *map2, int Noffset);
int HYPRE_LSI_DDIlutDecomposeNew(HYPRE_LSI_DDIlut *ilut_ptr,MH_Matrix *Amat,
								 int total_recv_leng, int *recv_lengths, int *ext_ja, double *ext_aa, 
								 int *map, int *map2, int Noffset);
	
/* hypre_lsi_misc.c */
void HYPRE_LSI_Get_IJAMatrixFromFile(double **val, int **ia, 
									 int **ja, int *N, double **rhs, char *matfile, char *rhsfile);
int HYPRE_LSI_Search(int *list,int value,int list_length);
int HYPRE_LSI_Search2(int key, int nlist, int *list);
int HYPRE_LSI_GetParCSRMatrix(HYPRE_IJMatrix Amat, int nrows, int nnz, 
                              int *ia_ptr, int *ja_ptr, double *a_ptr) ;
void HYPRE_LSI_qsort1a( int *ilist, int *ilist2, int left, int right);
int HYPRE_LSI_SplitDSort2(double *dlist, int nlist, int *ilist, int limit);
int HYPRE_LSI_SplitDSort(double *dlist, int nlist, int *ilist, int limit);
int HYPRE_LSI_SolveIdentity(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                            HYPRE_ParVector b, HYPRE_ParVector x);
int HYPRE_LSI_Cuthill(int n, int *ia, int *ja, double *aa, int *order_array,
                      int *reorder_array);
int HYPRE_LSI_MatrixInverse( double **Amat, int ndim, double ***Cmat );
int HYPRE_LSI_PartitionMatrix( int nRows, int startRow, int *rowLengths,
                               int **colIndices, double **colValues,
                               int *nLabels, int **labels);


/* HYPRE_LSI_poly.c */
int HYPRE_LSI_PolyCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_LSI_PolyDestroy( HYPRE_Solver solver );
int HYPRE_LSI_PolySetOrder( HYPRE_Solver solver, int order);
int HYPRE_LSI_PolySetOutputLevel( HYPRE_Solver solver, int level);
int HYPRE_LSI_PolySolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b,   HYPRE_ParVector x );
int HYPRE_LSI_PolySetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b,   HYPRE_ParVector x );

/* HYPRE_LSI_schwarz.c */
int HYPRE_LSI_SchwarzCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_LSI_SchwarzDestroy( HYPRE_Solver solver );
int HYPRE_LSI_SchwarzSetBlockSize( HYPRE_Solver solver, int blksize);
int HYPRE_LSI_SchwarzSetNBlocks( HYPRE_Solver solver, int nblks);
int HYPRE_LSI_SchwarzSetILUTFillin( HYPRE_Solver solver, double fillin);
int HYPRE_LSI_SchwarzSetOutputLevel( HYPRE_Solver solver, int level);
int HYPRE_LSI_SchwarzSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
							HYPRE_ParVector b,   HYPRE_ParVector x );
int HYPRE_LSI_SchwarzSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
							HYPRE_ParVector b,   HYPRE_ParVector x );

/* HYPRE_parcsr_bicgs.c */
int HYPRE_ParCSRBiCGSCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_ParCSRBiCGSDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRBiCGSSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
						   HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
						   HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSSetTol( HYPRE_Solver solver, double tol );
int HYPRE_ParCSRBiCGSSetMaxIter( HYPRE_Solver solver, int max_iter );
int HYPRE_ParCSRBiCGSSetStopCrit( HYPRE_Solver solver, int stop_crit );
int HYPRE_ParCSRBiCGSSetPrecond( HYPRE_Solver  solver,
								 int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
													  HYPRE_ParVector b, HYPRE_ParVector x),
								 int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
													  HYPRE_ParVector b, HYPRE_ParVector x),
								 void                *precond_data );
int HYPRE_ParCSRBiCGSSetLogging( HYPRE_Solver solver, int logging);
int HYPRE_ParCSRBiCGSGetNumIterations(HYPRE_Solver solver,
									  int *num_iterations);
int HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm(HYPRE_Solver solver,
												  double *norm );

/* HYPRE_parcsr_bicgs.c */
int HYPRE_ParCSRBiCGSTABLCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_ParCSRBiCGSTABLDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRBiCGSTABLSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSTABLSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRBiCGSTABLSetTol( HYPRE_Solver solver, double tol );
int HYPRE_ParCSRBiCGSTABLSetSize( HYPRE_Solver solver, int size );
int HYPRE_ParCSRBiCGSTABLSetMaxIter( HYPRE_Solver solver, int max_iter );
int HYPRE_ParCSRBiCGSTABLSetStopCrit( HYPRE_Solver solver, int stop_crit );
int HYPRE_ParCSRBiCGSTABLSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void               *precond_data );
int HYPRE_ParCSRBiCGSTABLSetLogging( HYPRE_Solver solver, int logging);
int HYPRE_ParCSRBiCGSTABLGetNumIterations(HYPRE_Solver solver,
                                                 int *num_iterations);
int HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                       double *norm );

/* HYPRE_parcsr_fgmres.h */
int HYPRE_ParCSRFGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_ParCSRFGMRESDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRFGMRESSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRFGMRESSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRFGMRESSetKDim(HYPRE_Solver solver, int kdim);
int HYPRE_ParCSRFGMRESSetTol(HYPRE_Solver solver, double tol);
int HYPRE_ParCSRFGMRESSetMaxIter(HYPRE_Solver solver, int max_iter);
int HYPRE_ParCSRFGMRESSetStopCrit(HYPRE_Solver solver, int stop_crit);
int HYPRE_ParCSRFGMRESSetPrecond(HYPRE_Solver  solver,
          int (*precond)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void *precond_data);
int HYPRE_ParCSRFGMRESSetLogging(HYPRE_Solver solver, int logging);
int HYPRE_ParCSRFGMRESGetNumIterations(HYPRE_Solver solver,
                                              int *num_iterations);
int HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                          double *norm );
int HYPRE_ParCSRFGMRESUpdatePrecondTolerance(HYPRE_Solver  solver,
                             int (*set_tolerance)(HYPRE_Solver sol, double));

/* HYPRE_parcsr_lsicg.c */
int HYPRE_ParCSRLSICGCreate(MPI_Comm comm, HYPRE_Solver *solver);
int HYPRE_ParCSRLSICGDestroy(HYPRE_Solver solver);
int HYPRE_ParCSRLSICGSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRLSICGSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRLSICGSetTol(HYPRE_Solver solver, double tol);
int HYPRE_ParCSRLSICGSetMaxIter(HYPRE_Solver solver, int max_iter);
int HYPRE_ParCSRLSICGSetStopCrit(HYPRE_Solver solver, int stop_crit);
int HYPRE_ParCSRLSICGSetPrecond(HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void *precond_data );
int HYPRE_ParCSRLSICGSetLogging(HYPRE_Solver solver, int logging);
int HYPRE_ParCSRLSICGGetNumIterations(HYPRE_Solver solver,
                                             int *num_iterations);
int HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                         double *norm );

/* HYPRE_parcsr_maxwell.c */
int HYPRE_ParCSRCotreeCreate(MPI_Comm comm, HYPRE_Solver *solver);
int HYPRE_ParCSRCotreeDestroy(HYPRE_Solver solver);
int HYPRE_ParCSRCotreeSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b, HYPRE_ParVector x);
int HYPRE_ParCSRCotreeSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
							HYPRE_ParVector b, HYPRE_ParVector x);
int HYPRE_ParCSRCotreeSetTol(HYPRE_Solver solver, double tol);
int HYPRE_ParCSRCotreeSetMaxIter(HYPRE_Solver solver, int max_iter);	

/* HYPRE_parcsr_superlu.c */
int HYPRE_ParCSR_SuperLUCreate(MPI_Comm comm, HYPRE_Solver *solver);
int HYPRE_ParCSR_SuperLUDestroy(HYPRE_Solver solver);
int HYPRE_ParCSR_SuperLUSetOutputLevel(HYPRE_Solver solver, int);
int HYPRE_ParCSR_SuperLUSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
							  HYPRE_ParVector b,HYPRE_ParVector x);
int HYPRE_ParCSR_SuperLUSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
							  HYPRE_ParVector b, HYPRE_ParVector x);
	
/* HYPRE_parcsr_symqmr.c */
int HYPRE_ParCSRSymQMRCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_ParCSRSymQMRDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRSymQMRSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
							HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRSymQMRSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
							HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRSymQMRSetTol( HYPRE_Solver solver, double tol );
int HYPRE_ParCSRSymQMRSetMaxIter( HYPRE_Solver solver, int max_iter );
int HYPRE_ParCSRSymQMRSetStopCrit( HYPRE_Solver solver, int stop_crit );
int HYPRE_ParCSRSymQMRSetPrecond( HYPRE_Solver  solver,
								  int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
													   HYPRE_ParVector b, HYPRE_ParVector x),
								  int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
													   HYPRE_ParVector b, HYPRE_ParVector x),
								  void                *precond_data );
int HYPRE_ParCSRSymQMRSetLogging( HYPRE_Solver solver, int logging);
int HYPRE_ParCSRSymQMRGetNumIterations(HYPRE_Solver solver,
									   int *num_iterations);
int HYPRE_ParCSRSymQMRGetFinalRelativeResidualNorm(HYPRE_Solver solver,
												   double *norm );

/* HYPRE_parcsr_TFQmr.c */
int HYPRE_ParCSRTFQmrCreate( MPI_Comm comm, HYPRE_Solver *solver );
int HYPRE_ParCSRTFQmrDestroy( HYPRE_Solver solver );
int HYPRE_ParCSRTFQmrSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRTFQmrSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x );
int HYPRE_ParCSRTFQmrSetTol( HYPRE_Solver solver, double tol );
int HYPRE_ParCSRTFQmrSetMaxIter( HYPRE_Solver solver, int max_iter );
int HYPRE_ParCSRTFQmrSetStopCrit( HYPRE_Solver solver, int stop_crit );
int HYPRE_ParCSRTFQmrSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void               *precond_data );
int HYPRE_ParCSRTFQmrSetLogging( HYPRE_Solver solver, int logging);
int HYPRE_ParCSRTFQmrGetNumIterations(HYPRE_Solver solver,
                                                 int *num_iterations);
int HYPRE_ParCSRTFQmrGetFinalRelativeResidualNorm(HYPRE_Solver solver,
                                                       double *norm );

#ifdef __cplusplus
}
#endif

#endif
