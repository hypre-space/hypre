/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/* **************************************************************************** 
 * -- SuperLU routine (version 1.1) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center, 
 * and Lawrence Berkeley National Lab.
 * ************************************************************************* */

#ifdef MLI_SUPERLU

#include <string.h>
#include <strings.h>
#include "base/mli_defs.h"
#include "mli_solver_seqsuperlu.h"

/* ****************************************************************************
 * constructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SeqSuperLU::MLI_Solver_SeqSuperLU(char *name) : MLI_Solver(name)
{
   permR_      = NULL;
   permC_      = NULL;
   mliAmat_    = NULL;
   factorized_ = 0;
   localNRows_ = 0;
}

/* ****************************************************************************
 * destructor 
 * --------------------------------------------------------------------------*/

MLI_Solver_SeqSuperLU::~MLI_Solver_SeqSuperLU()
{
   if ( permR_ != NULL ) 
   {
      Destroy_SuperNode_Matrix(&superLU_Lmat);
      Destroy_CompCol_Matrix(&superLU_Umat);
   }
   if ( permR_ != NULL ) delete [] permR_;
   if ( permC_ != NULL ) delete [] permC_;
}

/* ****************************************************************************
 * setup 
 * --------------------------------------------------------------------------*/

int MLI_Solver_SeqSuperLU::setup( MLI_Matrix *Amat )
{
   int      nrows, nnz, *csrIA, *csrJA, *cscJA, *cscIA;
   int      irow, icol, *countArray, colNum, index;
   int      *etree, permcSpec, lwork, panelSize, relax, info;
   double   *csrAA, *cscAA, diagPivotThresh, dropTol;
   char     refact[1];
   hypre_CSRMatrix      *hypreA;
   SuperMatrix          AC;
   extern SuperLUStat_t SuperLUStat;

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/

   mliAmat_ = Amat;
   if ( strcmp( mliAmat_->getName(), "HYPRE_CSR" ) )
   {
      printf("MLI_Solver_SeqSuperLU::setup ERROR - not HYPRE_CSR.\n");
      exit(1);
   }
   hypreA = (hypre_CSRMatrix *) mliAmat_->getMatrix();

   /* ---------------------------------------------------------------
    * fetch matrix
    * -------------------------------------------------------------*/
 
   csrAA = hypre_CSRMatrixData(hypreA);
   csrIA = hypre_CSRMatrixI(hypreA);
   csrJA = hypre_CSRMatrixJ(hypreA);
   nrows = hypre_CSRMatrixNumRows(hypreA);
   nnz   = hypre_CSRMatrixNumNonzeros(hypreA);
   localNRows_ = nrows;

   /* ---------------------------------------------------------------
    * conversion from CSR to CSC 
    * -------------------------------------------------------------*/

   countArray = new int[nrows];
   for ( irow = 0; irow < nrows; irow++ ) countArray[irow] = 0;
   for ( irow = 0; irow < nrows; irow++ ) 
      for ( icol = csrIA[irow]; icol < csrIA[irow+1]; icol++ ) 
         countArray[csrJA[icol]]++;
   cscJA = (int *)    malloc( (nrows+1) * sizeof(int) );
   cscIA = (int *)    malloc( nnz * sizeof(int) );
   cscAA = (double *) malloc( nnz * sizeof(double) );
   cscJA[0] = 0;
   nnz = 0;
   for ( icol = 1; icol <= nrows; icol++ ) 
   {
      nnz += countArray[icol-1]; 
      cscJA[icol] = nnz;
   }
   for ( irow = 0; irow < nrows; irow++ )
   {
      for ( icol = csrIA[irow]; icol < csrIA[irow+1]; icol++ ) 
      {
         colNum = csrJA[icol];
         index  = cscJA[colNum]++;
         cscIA[index] = irow;
         cscAA[index] = csrAA[icol];
      }
   }
   cscJA[0] = 0;
   nnz = 0;
   for ( icol = 1; icol <= nrows; icol++ ) 
   {
      nnz += countArray[icol-1]; 
      cscJA[icol] = nnz;
   }
   delete [] countArray;

   /* ---------------------------------------------------------------
    * make SuperMatrix 
    * -------------------------------------------------------------*/
   
   dCreate_CompCol_Matrix(&superLU_Amat, nrows, nrows, cscJA[nrows], 
                          cscAA, cscIA, cscJA, NC, D_D, GE);
   *refact = 'N';
   etree   = new int[nrows];
   permC_  = new int[nrows];
   permR_  = new int[nrows];
   permcSpec = 0;
   get_perm_c(permcSpec, &superLU_Amat, permC_);
   sp_preorder(refact, &superLU_Amat, permC_, etree, &AC);
   diagPivotThresh = 1.0;
   dropTol = 0.0;
   panelSize = sp_ienv(1);
   relax = sp_ienv(2);
   StatInit(panelSize, relax);
   lwork = 0;
   dgstrf(refact, &AC, diagPivotThresh, dropTol, relax, panelSize,
          etree,NULL,lwork,permR_,permC_,&superLU_Lmat,&superLU_Umat,&info);
   Destroy_CompCol_Permuted(&AC);
   Destroy_CompCol_Matrix(&superLU_Amat);
   delete [] etree;
   factorized_ = 1;
   StatFree();
   return 0;
}

/* ****************************************************************************
 * This subroutine calls the SuperLU subroutine to perform LU 
 * backward substitution 
 * --------------------------------------------------------------------------*/

int MLI_Solver_SeqSuperLU::solve( MLI_Vector *fIn, MLI_Vector *uIn )
{
   int             irow, nrows, info;
   char            trans[1];
   hypre_Vector    *f, *u;
   double          *uData, *fData;
   SuperMatrix     B;

   /* -------------------------------------------------------------
    * check that the factorization has been called
    * -----------------------------------------------------------*/

   if ( ! factorized_ )
   {
      printf("MLI_Solver_SeqSuperLU::Solve ERROR - not factorized yet.\n");
      exit(1);
   }

   /* -------------------------------------------------------------
    * fetch matrix and vector parameters
    * -----------------------------------------------------------*/

   nrows  = localNRows_;
   u      = (hypre_Vector *) uIn->getVector();
   uData  = hypre_VectorData(u);
   f      = (hypre_Vector *) fIn->getVector();
   fData  = hypre_VectorData(f);
   for ( irow = 0; irow < nrows; irow++ ) uData[irow] = fData[irow];

   /* -------------------------------------------------------------
    * collect global vector and create a SuperLU dense matrix
    * -----------------------------------------------------------*/

   dCreate_Dense_Matrix(&B, nrows, 1, uData, nrows, DN, D_D, GE);

   /* -------------------------------------------------------------
    * solve the problem
    * -----------------------------------------------------------*/

   *trans  = 'N';
   dgstrs (trans, &superLU_Lmat, &superLU_Umat, permR_, permC_, &B, &info);

   /* -------------------------------------------------------------
    * clean up 
    * -----------------------------------------------------------*/

   Destroy_SuperMatrix_Store(&B);

   return info;
}

#endif

