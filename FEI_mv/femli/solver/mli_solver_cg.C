/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

#include "base/mli_defs.h"
#include "parcsr_mv/parcsr_mv.h"
#include "solver/mli_solver_cg.h"
#include "solver/mli_solver_jacobi.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_CG::MLI_Solver_CG() : MLI_Solver(MLI_SOLVER_CG_ID)
{
   Amat_             = NULL;
   rVec_             = NULL;
   zVec_             = NULL;
   pVec_             = NULL;
   apVec_            = NULL;
   maxIterations_    = 3;
   tolerance_        = 0.0;
   baseSolver_       = NULL;
   baseMethod_       = MLI_SOLVER_JACOBI_ID;
   zeroInitialGuess_ = 0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_CG::~MLI_Solver_CG()
{
   if ( rVec_  != NULL ) delete rVec_;
   if ( zVec_  != NULL ) delete zVec_;
   if ( pVec_  != NULL ) delete pVec_;
   if ( apVec_ != NULL ) delete apVec_;
   if ( baseSolver_ != NULL ) delete baseSolver_;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::setup(MLI_Matrix *Amat_in)
{
   int  numSweeps;
   char paramString[100], *argv[1];;
   hypre_ParCSRMatrix *A;

   /*-----------------------------------------------------------------
    * set local matrix
    *-----------------------------------------------------------------*/

   Amat_ = Amat_in;

   /*-----------------------------------------------------------------
    * set up preconditioner
    *-----------------------------------------------------------------*/

   switch( baseMethod_ )
   {
      case MLI_SOLVER_JACOBI_ID : baseSolver_ = new MLI_Solver_Jacobi();
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 2;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      default : printf("MLI_Solver_CG ERROR : no base method.\n");
                exit(1);
   }
   baseSolver_->setup(Amat_);
 
   /*-----------------------------------------------------------------
    * build auxiliary vectors
    *-----------------------------------------------------------------*/

   rVec_  = Amat_->createVector();
   zVec_  = Amat_->createVector();
   pVec_  = Amat_->createVector();
   apVec_ = Amat_->createVector();

   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   int                i, iter, localNRows;
   double             *pData, *zData;
   double             rho, rhom1, alpha, beta, sigma, rnorm;
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *ADiag;
   hypre_ParVector    *f, *u, *p, *z, *r, *ap;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A          = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   ADiag      = hypre_ParCSRMatrixDiag(A);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   f          = (hypre_ParVector *) f_in->getVector();
   u          = (hypre_ParVector *) u_in->getVector();
   
   /*-----------------------------------------------------------------
    * set up for outer iterations 
    *-----------------------------------------------------------------*/

   r  = (hypre_ParVector *) rVec_->getVector();  /* -- r  -- */
   z  = (hypre_ParVector *) zVec_->getVector();  /* -- z  -- */
   p  = (hypre_ParVector *) pVec_->getVector();  /* -- p  -- */
   ap = (hypre_ParVector *) apVec_->getVector(); /* -- ap -- */
   hypre_ParVectorCopy(f, r); /* -- r = f - A u -- */
   if (zeroInitialGuess_==0) hypre_ParCSRMatrixMatvec(-1.0,A,u,1.0,r);
   zeroInitialGuess_ = 0;
   if ( tolerance_ != 0.0 )
        rnorm = sqrt(hypre_ParVectorInnerProd(r, r));
   else rnorm = 1.0;

   /*-----------------------------------------------------------------
    * fetch auxiliary vectors
    *-----------------------------------------------------------------*/

   pData  = hypre_VectorData(hypre_ParVectorLocalVector(p));
   zData  = hypre_VectorData(hypre_ParVectorLocalVector(z));

   /*-----------------------------------------------------------------
    * Perform Jacobi iterations
    *-----------------------------------------------------------------*/
 
   iter = 0;
   rho  = 0.0;
   while ( iter < maxIterations_ && rnorm > tolerance_ )
   {
      iter++;
      hypre_ParVectorSetConstantValues(z, 0.0);
      baseSolver_->solve( rVec_, zVec_ );
      rhom1 = rho;
      rho   = hypre_ParVectorInnerProd(r, z);
      if ( iter == 1 ) 
      {
         beta = 0.0;
         hypre_ParVectorCopy(z, p);
      }
      else
      {
         beta = rho / rhom1;

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
         for ( i = 0; i < localNRows; i++ ) 
            pData[i] = beta * pData[i] + zData[i];

      }
      hypre_ParCSRMatrixMatvec(1.0, A, p, 0.0, ap);
      sigma = hypre_ParVectorInnerProd(p, ap);
      alpha = rho /sigma;
      hypre_ParVectorAxpy(alpha, p, u);  /* u = u + alpha p */
      hypre_ParVectorAxpy(-alpha, ap, r); /* r = r - alpha ap */
      if (tolerance_ != 0.0 && maxIterations_ > 1) 
         rnorm = sqrt(hypre_ParVectorInnerProd(r, r));
   }
   return(0); 
}

/******************************************************************************
 * set CG parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_CG::setParams( char *param_string, int argc, char **argv )
{
   char   param1[200];

   if ( !strcasecmp(param_string, "maxIterations") )
   {
      sscanf(param_string, "%s %d", param1, &maxIterations_);
      return 0;
   }
   else if ( !strcasecmp(param_string, "tolerance") )
   {
      sscanf(param_string, "%s %lg", param1, &tolerance_);
      return 0;
   }
   else if ( !strcasecmp(param_string, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
      return 0;
   }
   else
   {   
      printf("MLI_Solver_CG::setParams - parameter not recognized.\n");
      printf("                Params = %s\n", param_string);
      return 1;
   }
   return 0;
}

