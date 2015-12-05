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
 * $Revision: 1.6 $
 ***********************************************************************EHEADER*/




#ifndef __MLI_SOLVER_JACOBI_H__
#define __MLI_SOLVER_JACOBI_H__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the Damped Jacobi relaxation scheme
 *---------------------------------------------------------------------------*/

class MLI_Solver_Jacobi : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   int         nSweeps_;
   double      *relaxWeights_;
   double      *diagonal_;
   double      maxEigen_;
   MLI_Vector  *auxVec_;
   MLI_Vector  *auxVec2_;
   MLI_Vector  *auxVec3_;
   int         zeroInitialGuess_;
   int         numFpts_;
   int         *FptList_;
   int         ownAmat_;
   int         modifiedD_;

public :

   MLI_Solver_Jacobi(char *name);
   ~MLI_Solver_Jacobi();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *paramString, int argc, char **argv );
   int setParams( int ntimes, double *relax_weights );
   int getParams( char *paramString, int *argc, char **argv );
};

#endif

