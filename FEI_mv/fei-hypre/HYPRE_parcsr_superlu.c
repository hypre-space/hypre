/*BHEADER**********************************************************************
 * (c) 2004   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParCSR_SuperLU interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/utilities.h"
#include "HYPRE.h"
#include "parcsr_mv/parcsr_mv.h"
#include "parcsr_ls/parcsr_ls.h"

/*---------------------------------------------------------------------------
 * SUPERLU include files
 *-------------------------------------------------------------------------*/

#ifdef HAVE_SUPERLU
#include "dsp_defs.h"
#include "superlu_util.h"

typedef struct HYPRE_SuperLU_Struct
{
   int          factorized_;
   int          *permR_;
   int          *permC_;
   SuperMatrix  SLU_Lmat;
   SuperMatrix  SLU_Umat;
   int          outputLevel_;
}
HYPRE_SuperLU;
#endif

/***************************************************************************
 * HYPRE_ParCSR_SuperLUCreate - Return a SuperLU object "solver".  
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSR_SuperLUCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
#ifdef HAVE_SUPERLU
   int           nprocs;
   HYPRE_SuperLU *sluPtr;
   
   MPI_Comm_size(comm, &nprocs);
   if ( nprocs > 1 )
   {
      printf("HYPRE_ParCSR_SuperLUCreate ERROR - too many processors.\n");
      return -1;
   }
   sluPtr = (HYPRE_SuperLU *) malloc(sizeof(HYPRE_SuperLU));
   assert ( sluPtr != NULL );
   sluPtr->factorized_  = 0;
   sluPtr->permR_       = NULL;
   sluPtr->permC_       = NULL;
   sluPtr->outputLevel_ = 0;
   *solver = (HYPRE_Solver) sluPtr;
   return 0;
#else
   printf("HYPRE_ParCSR_SuperLUCreate ERROR - SuperLU not enabled.\n");
   *solver = (HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * HYPRE_ParCSR_SuperLUDestroy - Destroy a SuperLU object.
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSR_SuperLUDestroy( HYPRE_Solver solver )
{
#ifdef HAVE_SUPERLU
   HYPRE_SuperLU *sluPtr = (HYPRE_SuperLU *) solver;
   assert ( sluPtr != NULL );
   if ( sluPtr->permR_ != NULL ) StatFree();
   if ( sluPtr->permR_ != NULL ) free(sluPtr->permR_);
   if ( sluPtr->permC_ != NULL ) free(sluPtr->permC_);
   free(sluPtr);
   return 0;
#else
   printf("HYPRE_ParCSR_SuperLUDestroy ERROR - SuperLU not enabled.\n");
   *solver = (HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * HYPRE_ParCSR_SuperLUSetOutputLevel - Set debug level 
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSR_SuperLUSetOutputLevel(HYPRE_Solver solver, int level)
{
#ifdef HAVE_SUPERLU
   HYPRE_SuperLU *sluPtr = (HYPRE_SuperLU *) solver;
   assert ( sluPtr != NULL );
   sluPtr->outputLevel_ = level;
   return 0;
#else
   printf("HYPRE_ParCSR_SuperLUSetOutputLevel ERROR - SuperLU not enabled.\n");
   *solver = (HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * HYPRE_ParCSR_SuperLUSetup - Set up function for SuperLU.
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSR_SuperLUSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A_csr,
                              HYPRE_ParVector b, HYPRE_ParVector x )
{
#ifdef HAVE_SUPERLU
   int    startRow, endRow, nrows, *partition, *AdiagI, *AdiagJ, nnz;
   int    irow, colNum, index, *cscI, *cscJ, jcol, *colLengs;
   int    *etree, permcSpec, lwork, panelSize, relax, info;
   double *AdiagA, *cscA, diagPivotThresh, dropTol;
   char    refact[1];
   hypre_CSRMatrix *Adiag;
   HYPRE_SuperLU   *sluPtr;
   SuperMatrix     sluAmat, auxAmat;

   /* ---------------------------------------------------------------- */
   /* get matrix information                                           */
   /* ---------------------------------------------------------------- */

   sluPtr = (HYPRE_SuperLU *) solver;
   assert ( sluPtr != NULL );
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
   startRow = partition[0];
   endRow   = partition[1] - 1;
   nrows    = endRow - startRow + 1;
   free( partition );
   if ( startRow != 0 )
   {
      printf("HYPRE_ParCSR_SuperLUSetup ERROR - start row != 0.\n");
      return -1;
   }

   /* ---------------------------------------------------------------- */
   /* get hypre matrix                                                 */
   /* ---------------------------------------------------------------- */

   Adiag  = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *) A_csr);
   AdiagI = hypre_CSRMatrixI(Adiag);
   AdiagJ = hypre_CSRMatrixJ(Adiag);
   AdiagA = hypre_CSRMatrixData(Adiag);
   nnz    = AdiagI[nrows];

   /* ---------------------------------------------------------------- */
   /* convert the csr matrix into csc matrix                           */
   /* ---------------------------------------------------------------- */

   colLengs = (int *) malloc(nrows * sizeof(int));
   for ( irow = 0; irow < nrows; irow++ ) colLengs[irow] = 0;
   for ( irow = 0; irow < nrows; irow++ )
      for ( jcol = AdiagI[irow]; jcol < AdiagI[irow+1]; jcol++ )
         colLengs[AdiagJ[jcol]]++;
   cscJ = (int *)    malloc( (nrows+1) * sizeof(int) );
   cscI = (int *)    malloc( nnz * sizeof(int) );
   cscA = (double *) malloc( nnz * sizeof(double) );
   cscJ[0] = 0;
   nnz = 0;
   for ( jcol = 1; jcol <= nrows; jcol++ )
   {
      nnz += colLengs[jcol-1];
      cscJ[jcol] = nnz;
   }
   for ( irow = 0; irow < nrows; irow++ )
   {
      for ( jcol = AdiagI[irow]; jcol < AdiagI[irow+1]; jcol++ )
      {
         colNum = AdiagJ[jcol];
         index  = cscJ[colNum]++;
         cscI[index] = irow;
         cscA[index] = AdiagA[jcol];
      }
   }
   cscJ[0] = 0;
   nnz = 0;
   for ( jcol = 1; jcol <= nrows; jcol++ )
   {
      nnz += colLengs[jcol-1];
      cscJ[jcol] = nnz;
   }
   free(colLengs);

   /* ---------------------------------------------------------------- */
   /* create SuperMatrix                                                */
   /* ---------------------------------------------------------------- */
                                                                                
   dCreate_CompCol_Matrix(&sluAmat,nrows,nrows,cscJ[nrows],cscA,cscI,
                          cscJ, NC, D_D, GE);
   *refact = 'N';
   etree   = (int *) malloc(nrows * sizeof(int));
   sluPtr->permC_  = (int *) malloc(nrows * sizeof(int));
   sluPtr->permR_  = (int *) malloc(nrows * sizeof(int));
   permcSpec = 0;
   get_perm_c(permcSpec, &sluAmat, sluPtr->permC_);
   sp_preorder(refact, &sluAmat, sluPtr->permC_, etree, &auxAmat);
   diagPivotThresh = 1.0;
   dropTol = 0.0;
   panelSize = sp_ienv(1);
   relax = sp_ienv(2);
   StatInit(panelSize, relax);
   lwork = 0;
   dgstrf(refact, &auxAmat, diagPivotThresh, dropTol, relax, panelSize,
          etree,NULL,lwork,sluPtr->permR_,sluPtr->permC_,&(sluPtr->SLU_Lmat),
          &(sluPtr->SLU_Umat),&info);
   Destroy_CompCol_Permuted(&auxAmat);
   Destroy_CompCol_Matrix(&sluAmat);
   free(etree);
   sluPtr->factorized_ = 1;
   return 0;
#else
   printf("HYPRE_ParCSR_SuperLUSetup ERROR - SuperLU not enabled.\n");
   *solver = (HYPRE_Solver) NULL;
   return -1;
#endif
}

/***************************************************************************
 * HYPRE_ParCSR_SuperLUSolve - Solve function for SuperLU.
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSR_SuperLUSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector b, HYPRE_ParVector x )
{
#ifdef HAVE_SUPERLU
   int    nrows, i, info;
   double *bData, *xData;
   char   trans[1];
   SuperMatrix B;
   HYPRE_SuperLU *sluPtr = (HYPRE_SuperLU *) solver;

   /* ---------------------------------------------------------------- */
   /* make sure setup has been called                                  */
   /* ---------------------------------------------------------------- */

   assert ( sluPtr != NULL );
   if ( ! (sluPtr->factorized_) )
   {
      printf("HYPRE_ParCSR_SuperLUSolve ERROR - not factorized yet.\n");
      return -1;
   }

   /* ---------------------------------------------------------------- */
   /* fetch right hand side and solution vector                        */
   /* ---------------------------------------------------------------- */

   xData = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *)x));
   bData = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *)b));
   nrows = hypre_ParVectorGlobalSize((hypre_ParVector *)x); 
   for (i = 0; i < nrows; i++) xData[i] = bData[i];

   /* ---------------------------------------------------------------- */
   /* solve                                                            */
   /* ---------------------------------------------------------------- */

   dCreate_Dense_Matrix(&B, nrows, 1, bData, nrows, DN, D_D,GE);
                                                                                
   /* -------------------------------------------------------------
    * solve the problem
    * -----------------------------------------------------------*/
                                                                                
   *trans  = 'N';
   dgstrs (trans, &(sluPtr->SLU_Lmat), &(sluPtr->SLU_Umat), 
           sluPtr->permR_, sluPtr->permC_, &B, &info);
   Destroy_SuperMatrix_Store(&B);
   return 0;
#else
   printf("HYPRE_ParCSR_SuperLUSolve ERROR - SuperLU not enabled.\n");
   *solver = (HYPRE_Solver) NULL;
   return -1;
#endif
}

