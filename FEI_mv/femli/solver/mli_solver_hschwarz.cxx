/*BHEADER**********************************************************************
 * (c) 2003   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <math.h>
#include <string.h>
#include "parcsr_mv/parcsr_mv.h"
#include "parcsr_ls/parcsr_ls.h"
#include "parcsr_ls/par_amg.h"
#include "solver/mli_solver_hschwarz.h"

/******************************************************************************
 * symmetric Gauss-Seidel relaxation scheme in BoomerAMG
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSchwarz::MLI_Solver_HSchwarz(char *name) : MLI_Solver(name)
{
   Amat_            = NULL;
   nSweeps_         = 1;
   relaxWeights_    = new double[1];
   relaxWeights_[0] = 1.0;
   mliVec_          = NULL;
   blkSize_         = 6;
   printRNorm_      = 0;
   smoother_        = NULL;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSchwarz::~MLI_Solver_HSchwarz()
{
   if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
   if ( mliVec_       != NULL ) delete mliVec_;
   if ( smoother_     != NULL ) HYPRE_SchwarzDestroy( smoother_ );
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSchwarz::setup(MLI_Matrix *mat)
{
   Amat_   = mat;
   mliVec_ = Amat_->createVector();
   calcOmega();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSchwarz::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   HYPRE_ParCSRMatrix A;
   HYPRE_ParVector    u, f;
   A = (HYPRE_ParCSRMatrix) Amat_->getMatrix();
   u = (HYPRE_ParVector) uIn->getVector();
   f = (HYPRE_ParVector) fIn->getVector();
   HYPRE_SchwarzSolve(smoother_, A, f, u);
   return 0;
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSchwarz::setParams( char *paramString, int argc, char **argv )
{
   int    i;
   double *weights;
   char   param1[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "numSweeps") )
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_HSchwarz::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = new double[nSweeps_];
      for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = 1.0;
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_HSchwarz::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nSweeps_ = *(int*)  argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( relaxWeights_ != NULL ) delete [] relaxWeights_;
      relaxWeights_ = NULL;
      if ( weights != NULL )
      {
         relaxWeights_ = new double[nSweeps_];
         for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = weights[i];
      }
   }
   else if ( !strcmp(param1, "printRNorm") )
   {
      printRNorm_ = 1;
   }
   else if ( !strcmp(param1, "blkSize") )
   {
      sscanf(paramString, "%s %d", param1, &blkSize_);
      if ( blkSize_ < 1 ) blkSize_ = 1;
   }
   else
   {   
      printf("MLI_Solver_HSchwarz::setParams - parameter not recognized.\n");
      printf("                 Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * calculate relax weight
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSchwarz::calcOmega()
{
   int                i, relaxType=6, relaxTypes[2], level=0, numCGSweeps=10;
   int                one=1, zero=0;
   double             relaxWt, dOne=1.0;
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *vTemp;
   hypre_ParAMGData   *amgData;
   HYPRE_Solver       *smoother;

   A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   amgData = (hypre_ParAMGData *) hypre_BoomerAMGCreate();
   amgData->A_array = new hypre_ParCSRMatrix*[1];
   amgData->A_array[0] = A;
   amgData->CF_marker_array = new int*[1];
   amgData->CF_marker_array[0] = NULL;
   relaxTypes[0] = 0;
   relaxTypes[1] = relaxType;
   amgData->grid_relax_type = relaxTypes;
   vTemp = (hypre_ParVector *) mliVec_->getVector();
   amgData->Vtemp = vTemp;

   amgData->smooth_type = relaxType;
   amgData->smooth_num_levels = 1;
   amgData->smooth_num_sweeps = one;

   smoother = hypre_CTAlloc(HYPRE_Solver, one);
   amgData->smoother = smoother;
   HYPRE_SchwarzCreate(&smoother[0]);
   HYPRE_SchwarzSetNumFunctions(smoother[0], blkSize_);
   HYPRE_SchwarzSetVariant(smoother[0], zero);
   HYPRE_SchwarzSetOverlap(smoother[0], one);
   HYPRE_SchwarzSetDomainType(smoother[0], one);
   HYPRE_SchwarzSetRelaxWeight(smoother[0], dOne);
   HYPRE_SchwarzSetup(smoother[0], (HYPRE_ParCSRMatrix) A, 
                      (HYPRE_ParVector) vTemp, (HYPRE_ParVector) vTemp);

   hypre_BoomerAMGCGRelaxWt((void *) amgData, level, numCGSweeps, &relaxWt);
   for ( i = 0; i < nSweeps_; i++ ) relaxWeights_[i] = relaxWt;
   printf("HSchwarz : relaxWt = %e\n", relaxWt);
   delete [] amgData->A_array;
   delete [] amgData->CF_marker_array;
   smoother_ = smoother[0];
   HYPRE_SchwarzSetRelaxWeight(smoother[0], relaxWt);
   hypre_TFree(amgData);
   return 0;
}

