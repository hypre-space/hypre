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
/* Data structure to hold the most basic information about a finite     */
/* element (used locally only).                                         */
/* ******************************************************************** */
/* Author        : Charles Tong (LLNL) and Raymond Tuminaro (SNL)       */
/* Date          : April, 1998                                          */
/* ******************************************************************** */

#ifndef _MLELMNTAGX_
#define _MLELMNTAGX_

#include <stdio.h>
/* #include <stdlib.h> */

#include "ml_common.h"
#include "ml_memory.h"

/* ******************************************************************* */
/*  ndim      : dimension of the grid                                  */
/*  Nvertices : number of vertices in the element                      */
/*  vertices  : an array storing the node number of the vertices       */
/*  x,y,z     : stores the coordinates of the vertices                 */
/* ------------------------------------------------------------------- */

typedef struct ML_ElementAGX_Struct
 {
   int          ndim;
   int          Nvertices;
   int          *vertices;
   double       *x, *y, *z;

} ML_ElementAGX;

/* ******************************************************************** */
/* functions to manipulate the Simple_element structure                 */
/* -------------------------------------------------------------------- */

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" 
{
#endif
#endif

extern int  ML_ElementAGX_Create(ML_ElementAGX**, int, int);
extern int  ML_ElementAGX_Destroy(ML_ElementAGX **);
extern int  ML_ElementAGX_Reuse(ML_ElementAGX *);
extern int  ML_ElementAGX_Print(ML_ElementAGX *);
extern int  ML_ElementAGX_Load_VertCoordinate
             (ML_ElementAGX*, int, double, double, double);
extern int  ML_ElementAGX_Get_VertCoordinate
             (ML_ElementAGX *, int, int*, double *, double *, double *);
extern int  ML_ElementAGX_ComposeCandidates
             (ML_ElementAGX *, int, double *, int *, int *, int *, int *);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif

