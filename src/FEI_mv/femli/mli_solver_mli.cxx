/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_parcsr_mv.h"
#include "mli_solver_mli.h"
#include "mli_method_amgsa.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_MLI::MLI_Solver_MLI(char *name) : MLI_Solver(name)
{
   Amat_ = NULL;
   mli_  = NULL;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_MLI::~MLI_Solver_MLI()
{
   Amat_ = NULL;
   if ( mli_ != NULL ) delete mli_;
}

/******************************************************************************
 * set up the solver
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLI::setup(MLI_Matrix *mat)
{
   int                iOne=1, level0=0, targc;
   double             dOne=1.0;
   char               *targv[2], paramString[100];
   MPI_Comm           comm;
   MLI_Method         *method;
   hypre_ParCSRMatrix *hypreA;

   Amat_  = mat;
   hypreA = (hypre_ParCSRMatrix *) Amat_->getMatrix();
   comm   = hypre_ParCSRMatrixComm( hypreA );
   if ( mli_ != NULL ) delete mli_;
   mli_ = new MLI(comm);
   method = new MLI_Method_AMGSA(comm);
   sprintf(paramString, "setMinCoarseSize 100");
   method->setParams( paramString, 0, NULL );
   targc    = 2;
   targv[0] = (char *) &iOne;
   targv[1] = (char *) &dOne;
   sprintf(paramString, "setPreSmoother SGS");
   method->setParams( paramString, targc, targv );
   mli_->setMethod(method);
   mli_->setSystemMatrix(level0, Amat_);
   mli_->setMaxIterations( 1 );
mli_->setOutputLevel( 2 );
   mli_->setup();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLI::solve(MLI_Vector *fIn, MLI_Vector *uIn)
{
   if ( mli_ == NULL )
   {
      printf("MLI_Solver_MLI::solve ERROR - no mli\n");
      exit(1);
   }
   mli_->solve( uIn, fIn );
   return 0;
}

