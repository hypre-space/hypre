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

#ifndef __MLSETUP__
#define __MLSETUP__


/* ******************************************************************** */
/* variable to pass to the local compute_basis_coefficients function    */
/* ******************************************************************** */

ML_GridFunc  *gridfcns_basis=NULL;

/* ******************************************************************** */
/* definition of local subroutines                                      */
/* ******************************************************************** */
#ifdef __cplusplus
extern "C" {
#endif

#include "ml_common.h"


void ML_compose_global_grid(     void           *c_grid, 
                                 ML_GridFunc    *cgrid_fcns,
                                 ML_GridAGX     **g_c_grid,
                                 ML_Comm        *comm); 
void ML_construct_RP0(           void           *c_grid, 
                                 ML_GridFunc    *cgrid_fcns,
                                 void           *f_grid, 
                                 ML_GridFunc    *fgrid_fcns,
                                 ML_GridAGX     *g_c_grid, 
                                 ML_OperatorAGX **xsfer_op,
                                 ML_Comm        *comm); 
int  ML_remote_grid_candidates(  void           *f_grid, 
                                 ML_GridFunc    *fgrid_fcns,
                                 ML_GridFunc    *cgrid_fcns,
                                 ML_GridAGX     *g_c_grid, 
                                 ML_IntList     *cand_list, 
                                 ML_OperatorAGX *xsfer_op,
                                 ML_Comm        *comm); 
void ML_exchange_candidates(     ML_IntList     *cand_list, 
                                 void           *f_grid, 
                                 ML_GridFunc    *fgrid_fcns,
                                 ML_GridAGX     *g_c_grid, 
                                 ML_CommInfoAGX *combuf,
                                 ML_Comm        *comm);
void ML_get_basis_functions_coef(ML_CommInfoAGX *combuf, 
                                 void           *c_grid,
                                 ML_GridFunc    *cgrid_fcns,
                                 ML_OperatorAGX *xsfer_op);
void ML_exchange_coefficients(   void           *c_grid,
				 ML_GridFunc    *cgrid_fcns,
                                 ML_CommInfoAGX *combuf, 
                                 ML_OperatorAGX *xsfer_op,
                                 ML_Comm        *comm);
void ML_construct_RP1(           void           *fgrid,
                                 ML_GridFunc    *fgrid_fcns,
                                 void           *cgrid,
                                 ML_GridFunc    *cgrid_fcns,
                                 ML_GridAGX     *g_c_grid,
                                 ML_CommInfoAGX *combuf,
                                 ML_OperatorAGX *xsfer_op,
                                 ML_Comm        *comm);



#ifdef __cplusplus
}
#endif

#endif
