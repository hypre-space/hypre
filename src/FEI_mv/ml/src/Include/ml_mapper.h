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
/* Declaration of the ML_Mapper structure                               */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL) and Raymond Tuminaro (SNL)       */
/* Date          : March, 1999                                          */
/* ******************************************************************** */

#ifndef __MLMAPPERH__
#define __MLMAPPERH__

#include "ml_common.h"
#include "ml_defs.h"
#include "ml_memory.h"

#ifdef WKC
#include <Epetra_Comm.h>
#include <Epetra_MultiVector.h>
#endif 

typedef struct ML_Mapper_Struct ML_Mapper;

/* ******************************************************************** */
/* data definition for the ML_Mapper Class                              */
/* ******************************************************************** */
/* -------------------------------------------------------------------- */
/* This data structure stores a mapping function that maps unknowns     */
/* between the grid and equation spaces (One is instantiated for grid   */
/* to equation space mapping, and the other for equation to grid space  */
/* mapping.                                                             */
/* -------------------------------------------------------------------- */

struct ML_Mapper_Struct {
   int   ML_id;
   void  *data;
   int   inlength, outlength;
   int   (*map)(void*, double *, double *);
};

/* ******************************************************************** */
/* function for accessing the ML_Mapper Class                           */
/* ******************************************************************** */

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

extern int ML_Mapper_Create(ML_Mapper **);
extern int ML_Mapper_Init(ML_Mapper *);
extern int ML_Mapper_Destroy(ML_Mapper **);
extern int ML_Mapper_Clean(ML_Mapper *);
extern int ML_Mapper_Check(ML_Mapper *);
extern int ML_Mapper_SetFunc(ML_Mapper*,int,int,
                         int (*func)(void*,double*,double*));
extern int ML_Mapper_SetData(ML_Mapper*,void*);
extern int ML_Mapper_GetLength(ML_Mapper*,int*,int*);
extern int ML_Mapper_Apply(ML_Mapper *, double *, double *);

/* WKC -- Added header for Epetra stuff */
#ifdef WKC
extern int ML_Mapper_Apply(ML_Mapper *, Epetra_MultiVector & , Epetra_MultiVector &);
#endif

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

