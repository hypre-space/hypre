/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
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
