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
#include "solver/mli_solver_hsgs.h"

/******************************************************************************
 * symmetric Gauss-Seidel relaxation scheme in BoomerAMG
 *****************************************************************************/

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSGS::MLI_Solver_HSGS(char *name) : MLI_Solver(name)
{
   Amat_         = NULL;
   nSweeps_      = 1;
   relaxWeights_ = 1.0;
   relaxOmega_   = 1.0;
   mliVec_       = NULL;
   calcOmega_    = 1;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_HSGS::~MLI_Solver_HSGS()
{
   if (mliVec_ != NULL) delete mliVec_;
   mliVec_ = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::setup(MLI_Matrix *mat)
{
   Amat_ = mat;
   if (mliVec_ != NULL) delete mliVec_;
   mliVec_ = Amat_->createVector();
   if (calcOmega_ == 1) calcOmega();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   int                relaxType=6, relaxPts=0, iS;
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *f, *u, *vTemp;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A     = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   u     = (hypre_ParVector *) uIn->getVector();
   f     = (hypre_ParVector *) fIn->getVector();
   vTemp = (hypre_ParVector *) mliVec_->getVector();
   for (iS = 0; iS < nSweeps_; iS++)
      hypre_BoomerAMGRelax(A,f,NULL,relaxType,relaxPts,relaxWeights_,
                           relaxOmega_,u,vTemp);
   return 0;
}

/******************************************************************************
 * set SGS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::setParams(char *paramString, int argc, char **argv)
{
   double *weights;
   char   param1[100];

   sscanf(paramString, "%s", param1);
   if (!strcmp(param1, "numSweeps"))
   {
      if ( argc != 1 ) 
      {
         printf("MLI_Solver_HSGS::setParams ERROR : needs 1 arg.\n");
         return 1;
      }
      nSweeps_ = *(int*) argv[0];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_HSGS::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nSweeps_ = *(int*)  argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      if ( nSweeps_ < 1 ) nSweeps_ = 1;
      if ( weights != NULL ) relaxWeights_ = weights[0]; 
   }
   else if ( !strcmp(param1, "calcOmega") )
   {
      calcOmega_ = 1;
   }
   else
   {   
      printf("MLI_Solver_HSGS::setParams - parameter not recognized.\n");
      printf("                 Params = %s\n", paramString);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * calculate relax weight
 *---------------------------------------------------------------------------*/

int MLI_Solver_HSGS::calcOmega()
{
   int                i, relaxType=6, relaxTypes[2], level=0, numCGSweeps=10;
   double             relaxOmega;
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *vTemp;
   hypre_ParAMGData   *amgData;

   A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   amgData = (hypre_ParAMGData *) hypre_BoomerAMGCreate();
   amgData->CF_marker_array = new int*[1];
   amgData->CF_marker_array[0] = NULL;
   amgData->A_array = new hypre_ParCSRMatrix*[1];
   amgData->A_array[0] = A;
   vTemp = (hypre_ParVector *) mliVec_->getVector();
   amgData->Vtemp = vTemp;
   relaxTypes[0] = 0;
   relaxTypes[1] = relaxType;
   amgData->grid_relax_type = relaxTypes;
   amgData->smooth_num_levels = 0;
   amgData->smooth_type = 0;
   hypre_BoomerAMGCGRelaxWt((void *) amgData,level,numCGSweeps,&relaxOmega_);
   printf("HYPRE/FEI/MLI HSGS : relaxOmega = %e\n", relaxOmega_);
   delete [] amgData->A_array;
   delete [] amgData->CF_marker_array;
   hypre_TFree(amgData);
   return 0;
}

