/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.8 $
 ***********************************************************************EHEADER*/





#ifndef __MLI_SOLVER_MLS_H__
#define __MLI_SOLVER_MLS_H__

#include <stdio.h>
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"

/******************************************************************************
 * data structure for the MLS smoother by Brezina S = (I - omega A)^2 A
 * where omega = 2 / max_eigen of A)
 *---------------------------------------------------------------------------*/

class MLI_Solver_MLS : public MLI_Solver
{
   MLI_Matrix  *Amat_;
   MLI_Vector  *Vtemp_;
   MLI_Vector  *Wtemp_;
   MLI_Vector  *Ytemp_;
   double      maxEigen_;
   int         mlsDeg_; /* degree for current level */
   double      mlsBoost_;
   double      mlsOver_;
   double      mlsOm_[5];
   double      mlsOm2_;
   double      mlsCf_[5];
   int         zeroInitialGuess_;

public :

   MLI_Solver_MLS(char *name);
   ~MLI_Solver_MLS();
   int setup(MLI_Matrix *Amat);
   int solve(MLI_Vector *f, MLI_Vector *u);

   int setParams( char *paramString, int argc, char **argv );
   int setParams( double maxEigen );
};

#endif

