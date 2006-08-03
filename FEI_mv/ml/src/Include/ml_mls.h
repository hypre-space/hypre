/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef __MLMLS__
#define __MLMLS__

#include "ml_common.h"

#define MLS_MAX_DEG  5 /* max. degree of MLS smoother        */

struct MLSthing {
 /*
  * set degree and 
  * the precomputed coefficients used in MLS application
  */
  int     mlsDeg; /* degree for current level */
  double  mlsBoost; 
  double  mlsOver; 
  double  mlsOm[MLS_MAX_DEG];
  double  mlsOm2;
  double  mlsCf[MLS_MAX_DEG];
  double *pAux, *res, *y;  /* workarrays allocated in .... to be reused */
  double eig_ratio;
  double beta_real, beta_img;

  ML_Sm_BGS_Data *block_scaling;/* these last arguments are used to */
  ML_Operator *unscaled_matrix;     /* implement block scaling instead  */
  ML_Operator *scaled_matrix;       /* of point scaling when doing      */
                                    /* Chebyshev smoothing. The basic   */
                                    /* idea is to turn off diagonal     */
                                    /* scaling by setting the diagonal  */
                                    /* to 1. Then to create a matrix    */
                                    /* wrapper that does Dinv*Amat*v    */
                                    /* when a matvec is requested (where*/
                                    /* D is some block diagonal */
};

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

int   ML_MLS_Smooth0( double b[], double vx[], double vy[], int deg, 
		      double *om, double *cf, int nit, double over, 
		      double wk);
int   ML_MLS_Smooth1( double b[], double vx[], double vy[], int deg, 
		      double *om, double *cf, int nit, double over, 
		      double wk);
int ML_MLS_SandwPres(void *sm, int inlen, double x[], int outlen, double y[]);
int ML_MLS_SandwPost(void *sm, int inlen, double x[], int outlen, double y[]);
int ML_MLS_SPrime_Apply(void *sm,int inlen,double x[],int outlen, double rhs[]);


#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif
