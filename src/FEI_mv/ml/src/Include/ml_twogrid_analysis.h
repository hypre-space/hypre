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
 * $Revision: 1.3 $
 ***********************************************************************EHEADER*/



/* ******************************************************************** */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */        
/* ******************************************************************** */

/* ******************************************************************** */
/* Some tools for two grid analysis.                                    */
/* ******************************************************************** */
/* Author        : Jonathan Hu (SNL)                                    */
/* Date          : September, 2001                                      */
/* ******************************************************************** */

#ifndef __MLTWOGRID__
#define __MLTWOGRID__

#include "ml_common.h"
#include "ml_defs.h"
#include "ml_struct.h"

#ifndef ML_CPP
#ifdef __cplusplus
   extern "C" {
#endif
#endif

extern double ML_gdot_H0(ML_Operator *Amat, double *vec1, double *vec2);
extern double ML_gdot_H1(ML_Operator *Amat, double *vec1, double *vec2);
extern double ML_gdot_H2(ML_Operator *Amat, double *vec1, double *vec2);
extern double ML_GetCoarseGridConst(ML_Operator *Amat, ML_Operator *Rmat,
                                    ML_Operator *Pmat, double *err_h);
extern double ML_GetSmoothingConst(ML_Operator *Amat, double *err_h,
                                    ML_Smoother *sm);
/*
extern double ML_GetTwoLevelConvergenceFactor(ML_Operator *Amat,
                                    ML_Operator *Rmat, ML_Operator *Pmat,
                                    ML_Smoother *sm,
                                    double *approx_soln, double *exact_soln);
*/
double ML_GetTwoLevelConvergenceFactor(ML *ml, double *approx_soln,
									   double *exact_soln);


#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif /*ifdef __MLTWOGRID__*/

