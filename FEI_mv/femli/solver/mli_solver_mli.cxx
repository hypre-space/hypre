/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "solver/mli_solver_mli.h"
#include "amgs/mli_method_amgsa.h"

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

