/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





#include <stdio.h>
#include <string.h>
#include "solver/mli_solver_gmres.h"
#include "solver/mli_solver_jacobi.h"
#include "solver/mli_solver_bjacobi.h"
#include "solver/mli_solver_bsgs.h"
#include "solver/mli_solver_hsgs.h"
#include "solver/mli_solver_mli.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_GMRES::MLI_Solver_GMRES(char *name) : MLI_Solver(name)
{
   Amat_ = NULL;
   KDim_ = 20;
   tolerance_ = 1.0e-16;
   maxIterations_ = 1000;
   rVec_ = NULL;
   pVec_ = NULL;
   zVec_ = NULL;
   baseSolver_ = NULL;
   baseMethod_ = MLI_SOLVER_SGS_ID;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_GMRES::~MLI_Solver_GMRES()
{
   int i;
   if ( rVec_  != NULL ) delete rVec_;
   if ( pVec_  != NULL ) 
   {
      for (i = 0; i < KDim_+1; i++) delete pVec_[i];
      delete [] pVec_;
   }
   if ( zVec_  != NULL ) 
   {
      for (i = 0; i < KDim_+1; i++) delete zVec_[i];
      delete [] zVec_;
   }
   if ( baseSolver_ != NULL ) delete baseSolver_;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_GMRES::setup(MLI_Matrix *Amat_in)
{
   int    i, numSweeps;
   double value=4.0/3.0;
   char   paramString[100], *argv[1];;

   /*-----------------------------------------------------------------
    * set local matrix
    *-----------------------------------------------------------------*/

   Amat_ = Amat_in;

   /*-----------------------------------------------------------------
    * set up preconditioner
    *-----------------------------------------------------------------*/

   if ( baseSolver_ != NULL ) delete baseSolver_;
   switch( baseMethod_ )
   {
      case MLI_SOLVER_JACOBI_ID : sprintf(paramString, "Jacobi");
                                  baseSolver_ = 
                                     new MLI_Solver_Jacobi(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  sprintf(paramString, "setMaxEigen");
                                  argv[0] = (char *) &value;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_BJACOBI_ID: sprintf(paramString, "BJacobi");
                                  baseSolver_ = 
                                     new MLI_Solver_BJacobi(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_SGS_ID :    sprintf(paramString, "HSGS");
                                  baseSolver_ = 
                                     new MLI_Solver_HSGS(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_BSGS_ID :   sprintf(paramString, "BSGS");
                                  baseSolver_ = 
                                     new MLI_Solver_BSGS(paramString);
                                  sprintf(paramString, "numSweeps");
                                  numSweeps = 1;
                                  argv[0] = (char *) &numSweeps;
                                  baseSolver_->setParams(paramString,1,argv);
                                  break;
      case MLI_SOLVER_MLI_ID :    sprintf(paramString, "MLI");
                                  baseSolver_ = 
                                     new MLI_Solver_BSGS(paramString);
                                  break;
      default : printf("MLI_Solver_GMRES ERROR : no base method.\n");
                exit(1);
   }
   baseSolver_->setup(Amat_);
 
   /*-----------------------------------------------------------------
    * destroy previously allocated auxiliary vectors
    *-----------------------------------------------------------------*/

   if ( rVec_ != NULL ) delete rVec_;
   if ( pVec_ != NULL ) 
   {
      for (i = 0; i < KDim_+1; i++) delete pVec_[i];
      delete [] pVec_;
   }
   if ( zVec_  != NULL ) 
   {
      for (i = 0; i < KDim_+1; i++) delete zVec_[i];
      delete [] zVec_;
   }

   /*-----------------------------------------------------------------
    * build auxiliary vectors
    *-----------------------------------------------------------------*/

   rVec_ = Amat_->createVector();
   pVec_ = new MLI_Vector*[KDim_+1];
   zVec_ = new MLI_Vector*[KDim_+1];
   for ( i = 0; i <= KDim_; i++ )
      pVec_[i] = Amat_->createVector();
   for ( i = 0; i <= KDim_; i++ )
      zVec_[i] = Amat_->createVector();

   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_GMRES::solve(MLI_Vector *b_in, MLI_Vector *u_in)
{
   hypre_ParCSRMatrix *A;
   hypre_ParVector    *b, *u, *r, **p, **z;
   int	              i, j, k, ierr = 0, iter, mypid;
   double             *rs, **hh, *c, *s, t, zero=0.0;
   double             epsilon, gamma1, rnorm, epsmac = 1.e-16; 
   char               paramString[100];
   MPI_Comm           comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix/vector parameters
    *-----------------------------------------------------------------*/

   A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   b = (hypre_ParVector *) b_in->getVector();
   u = (hypre_ParVector *) u_in->getVector();
   HYPRE_ParCSRMatrixGetComm((HYPRE_ParCSRMatrix) A, &comm);
   MPI_Comm_rank(comm, &mypid);
   
   /*-----------------------------------------------------------------
    * fetch other auxiliary vectors
    *-----------------------------------------------------------------*/

   r  = (hypre_ParVector *) rVec_->getVector();
   p  = (hypre_ParVector **) malloc(sizeof(hypre_ParVector *) * (KDim_+1)); 
   z  = (hypre_ParVector **) malloc(sizeof(hypre_ParVector *) * (KDim_+1)); 
   for ( i = 0; i <= KDim_; i++ )
      p[i] = (hypre_ParVector *) pVec_[i]->getVector();
   for ( i = 0; i <= KDim_; i++ )
      z[i] = (hypre_ParVector *) zVec_[i]->getVector();

   rs = new double[KDim_+1]; 
   c  = new double[KDim_]; 
   s  = new double[KDim_]; 
   hh = new double*[KDim_+1]; 
   for (i=0; i < KDim_+1; i++) hh[i] = new double[KDim_]; 

   /*-----------------------------------------------------------------
    * compute initial rnorm
    *-----------------------------------------------------------------*/

   hypre_ParVectorSetConstantValues(u, zero);
   hypre_ParVectorCopy(b, r);
   //hypre_ParCSRMatrixMatvec(-1.0,A,u,1.0,r);

   rnorm = sqrt(hypre_ParVectorInnerProd(r, r));
   if ( tolerance_ != 0.0 ) epsilon = rnorm * tolerance_;
   else                     epsilon = 1.0;

   /*-----------------------------------------------------------------
    * Perform iterations
    *-----------------------------------------------------------------*/
 
   hypre_ParVectorCopy(r,p[0]);
   iter = 0;
   strcpy( paramString, "zeroInitialGuess" );

   while (iter < maxIterations_)
   {
      rs[0] = rnorm;
      if (rnorm == 0.0)
      {
         ierr = 0;
         return ierr;
      }
      if (rnorm <= epsilon && iter > 0) 
      {
         hypre_ParVectorCopy(b,r);
         hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);
         rnorm = sqrt(hypre_ParVectorInnerProd(r,r));
         if (rnorm <= epsilon) break;
      }
      t = 1.0 / rnorm;
      hypre_ParVectorScale(t,p[0]);
      i = 0;
      while (i < KDim_ && rnorm > epsilon && iter < maxIterations_)
      {
         i++;
         iter++;
         hypre_ParVectorSetConstantValues(z[i-1], zero);

         baseSolver_->setParams( paramString, 0, NULL );
         baseSolver_->solve( pVec_[i-1], zVec_[i-1] );
         hypre_ParCSRMatrixMatvec(1.0, A, z[i-1], 0.0, p[i]);

         /* modified Gram_Schmidt */

         for (j=0; j < i; j++)
         {
            hh[j][i-1] = hypre_ParVectorInnerProd(p[j],p[i]);
            hypre_ParVectorAxpy(-hh[j][i-1],p[j],p[i]);
         }
         t = sqrt(hypre_ParVectorInnerProd(p[i],p[i]));
         hh[i][i-1] = t;	
         if (t != 0.0)
         {
            t = 1.0/t;
            hypre_ParVectorScale(t, p[i]);
         }

         /* done with modified Gram_schmidt. update factorization of hh */

         for (j = 1; j < i; j++)
         {
            t = hh[j-1][i-1];
            hh[j-1][i-1] = c[j-1]*t + s[j-1]*hh[j][i-1];		
            hh[j][i-1] = -s[j-1]*t + c[j-1]*hh[j][i-1];
         }
         gamma1 = sqrt(hh[i-1][i-1]*hh[i-1][i-1] + hh[i][i-1]*hh[i][i-1]);
         if (gamma1 == 0.0) gamma1 = epsmac;
         c[i-1] = hh[i-1][i-1]/gamma1;
         s[i-1] = hh[i][i-1]/gamma1;
         rs[i] = -s[i-1]*rs[i-1];
         rs[i-1] = c[i-1]*rs[i-1];

         /* determine residual norm */

         hh[i-1][i-1] = c[i-1]*hh[i-1][i-1] + s[i-1]*hh[i][i-1];
         rnorm = fabs(rs[i]);
      }

      /* now compute solution, first solve upper triangular system */
	
      rs[i-1] = rs[i-1]/hh[i-1][i-1];
      for (k = i-2; k >= 0; k--)
      {
         t = rs[k];
         for (j = k+1; j < i; j++) t -= hh[k][j]*rs[j];
         rs[k] = t/hh[k][k];
      }

	
      for (j = 0; j < i; j++) hypre_ParVectorAxpy(rs[j], z[j], u);

      /* check for convergence, evaluate actual residual */

      hypre_ParVectorCopy(b,p[0]);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, p[0]);
      rnorm = sqrt(hypre_ParVectorInnerProd(p[0],p[0]));
      if (mypid == -1) printf("GMRES iter = %d, rnorm = %e\n", iter, rnorm);
      if (rnorm <= epsilon) break;
   }

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   delete [] c;
   delete [] s;
   delete [] rs;
   for (i=0; i < KDim_+1; i++) delete [] hh[i];
   delete [] hh;
   free(p);
   free(z);
   return(0); 
}

/******************************************************************************
 * set GMRES parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_GMRES::setParams( char *paramString, int argc, char **argv )
{
   char   param1[100], param2[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "maxIterations") )
   {
      sscanf(paramString, "%s %d", param1, &maxIterations_);
      return 0;
   }
   else if ( !strcmp(param1, "tolerance") )
   {
      sscanf(paramString, "%s %lg", param1, &tolerance_);
      return 0;
   }
   else if ( !strcmp(param1, "numSweeps") )
   {
      sscanf(paramString, "%s %d", param1, &maxIterations_);
      return 0;
   }
   else if ( !strcmp(param1, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_GMRES::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) maxIterations_ = *(int*) argv[0];
      return 0;
   }
   else if ( !strcmp(param1, "baseMethod") )
   {
      sscanf(paramString, "%s %s", param1, param2);
      if ( !strcmp(param2, "Jacobi") ) 
         baseMethod_ = MLI_SOLVER_JACOBI_ID;
      else if ( !strcmp(param2, "BJacobi") )
         baseMethod_ = MLI_SOLVER_BJACOBI_ID;
      else if ( !strcmp(param2, "SGS") )
         baseMethod_ = MLI_SOLVER_SGS_ID;
      else if ( !strcmp(param2, "BSGS") )
         baseMethod_ = MLI_SOLVER_BSGS_ID;
      else if ( !strcmp(param2, "MLI") )
         baseMethod_ = MLI_SOLVER_MLI_ID;
      else
         baseMethod_ = MLI_SOLVER_BJACOBI_ID;
      return 0;
   }
   else
   {   
      printf("MLI_Solver_GMRES::setParams - parameter not recognized.\n");
      printf("                Params = %s\n", paramString);
      return 1;
   }
}

