/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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




/* ******************************************************************** */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */        
/* ******************************************************************** */

/* ******************************************************************** */
/* local (to ML) data structure to hold vector information              */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL)                                  */
/* Date          : August, 1997                                         */
/* ******************************************************************** */

#ifndef __MLVEC__
#define __MLVEC__

#include <stdio.h>
/* #include <stdlib.h> */
#include "ml_common.h"
#include "ml_memory.h"
#include "ml_comm.h"
#include "ml_defs.h"

/* ******************************************************************** */
/* definition of the grid structure                                     */
/*  ML_id             : identification for a vector                     */
/*  VecLength         : length of vector                                */
/*  SetOrLoad         : a flag to see if storage is allocated.          */
/*  VecData           : holder for data                                 */
/* -------------------------------------------------------------------- */

typedef struct ML_DVector_Struct
{
  int     ML_id;
  ML_Comm *comm;
  int     VecLength;
  int     SetOrLoad;
  double  *VecData;

} ML_DVector;

/* ******************************************************************** */
/* functions to manipulate the vector structure                         */
/* -------------------------------------------------------------------- */

#ifndef ML_CPP
#ifdef __cplusplus
extern "C"
{
#endif
#endif

extern int  ML_DVector_Create( ML_DVector **, ML_Comm *com );
extern int  ML_DVector_Init(ML_DVector *vec);
extern int  ML_DVector_Destroy( ML_DVector ** );
extern int  ML_DVector_Clean( ML_DVector *vec ); 
extern int  ML_DVector_LoadData( ML_DVector *, int, double * );
extern int  ML_DVector_SetData( ML_DVector *, int, double * );
extern int  ML_DVector_GetLength( ML_DVector * );
extern int  ML_DVector_GetData( ML_DVector *, int *, double * );
extern int  ML_DVector_GetDataPtr( ML_DVector *, double ** );
extern int  ML_DVector_Check( ML_DVector * );
extern int  ML_DVector_Scale( double, ML_DVector * );
extern int  ML_DVector_Copy( ML_DVector *, ML_DVector * );
extern int  ML_DVector_Axpy( double, ML_DVector *, ML_DVector * );
extern int  ML_DVector_Aypx( double, ML_DVector *, ML_DVector * );
extern int ML_DVector_Print(int length, double *data, char *label, ML_Comm *comm);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

