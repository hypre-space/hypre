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
/* Declaration of the ML_BdryPts structure                              */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL) and Raymond Tuminaro (SNL)       */
/* Date          : March, 1999                                          */
/* ******************************************************************** */

#ifndef __MLBDRYPTSH__
#define __MLBDRYPTSH__

#include "ml_common.h"
#include "ml_defs.h"
#include "ml_memory.h"

/* ******************************************************************** */
/* data definition for the ML_BdryPts Class                             */
/* ******************************************************************** */
/* -------------------------------------------------------------------- */
/* This data structure stores two arrays : one integer array for a list */
/* of boundary conditions in the grid space, and another integer array  */
/* for a list of boundary conditions in the equation space.  In this    */
/* implementation, only Dirichlet boundary conditions are stored.       */
/* -------------------------------------------------------------------- */

typedef struct ML_BdryPts_Struct ML_BdryPts;

struct ML_BdryPts_Struct {
   int    ML_id;
   int    Dirichlet_grid_CreateOrDup;
   int    Dirichlet_grid_length;
   int    *Dirichlet_grid_list;
   int    Dirichlet_eqn_CreateOrDup;
   int    Dirichlet_eqn_length;
   int    *Dirichlet_eqn_list;
};

/* ******************************************************************** */
/* function for accessing the ML_BdryPts Class                          */
/* ******************************************************************** */

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif 

extern int ML_BdryPts_Create(ML_BdryPts **);
extern int ML_BdryPts_Init(ML_BdryPts *);
extern int ML_BdryPts_Destroy(ML_BdryPts **);
extern int ML_BdryPts_Clean(ML_BdryPts *);
extern int ML_BdryPts_Check_Dirichlet_Grid(ML_BdryPts *);
extern int ML_BdryPts_Check_Dirichlet_Eqn(ML_BdryPts *);
extern int ML_BdryPts_Get_Dirichlet_Grid_Info(ML_BdryPts *, int *, int **);
extern int ML_BdryPts_Get_Dirichlet_Eqn_Info(ML_BdryPts *, int *, int **);
extern int ML_BdryPts_Load_Dirichlet_Grid(ML_BdryPts *,int,int*);
extern int ML_BdryPts_Load_Dirichlet_Eqn(ML_BdryPts *,int,int*);
extern int ML_BdryPts_Copy_Dirichlet_GridToEqn(ML_BdryPts *);
extern int ML_BdryPts_ApplyZero_Dirichlet_Grid(ML_BdryPts *, double *);
extern int ML_BdryPts_ApplyZero_Dirichlet_Eqn(ML_BdryPts *, double *);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

