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
/* Declaration of the Grid and its access functions data structure      */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL) and Raymond Tuminaro (SNL)       */
/* Date          : March, 1999                                          */
/* ******************************************************************** */

#ifndef __MLGRID__
#define __MLGRID__

#include "ml_common.h"
#include "ml_defs.h"
#include "ml_memory.h"
#include "ml_gridfunc.h"

typedef struct ML_Grid_Struct ML_Grid;

/* ******************************************************************** */
/* definition of the data structure for Grid                            */
/* -------------------------------------------------------------------- */

struct ML_Grid_Struct 
{
   int           ML_id;
   void          *Grid;        /* user grid data structure        */
   ML_GridFunc   *gridfcn;     /* a set of grid access functions  */
   int           gf_SetOrLoad; /* see if gridfcn is created locally */
};

/* ******************************************************************** */
/* definition of the functions                                          */
/* -------------------------------------------------------------------- */

#ifndef ML_CPP
#ifdef __cplusplus
extern "C"
{
#endif
#endif

extern int ML_Grid_Create( ML_Grid ** );
extern int ML_Grid_Init( ML_Grid * );
extern int ML_Grid_Destroy( ML_Grid ** );
extern int ML_Grid_Clean( ML_Grid * );
extern int ML_Grid_Set_Grid( ML_Grid *, void * );
extern int ML_Grid_Set_GridFunc( ML_Grid *, ML_GridFunc * );
extern int ML_Grid_Get_GridFunc( ML_Grid *, ML_GridFunc ** );
extern int ML_Grid_Create_GridFunc( ML_Grid * );

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

