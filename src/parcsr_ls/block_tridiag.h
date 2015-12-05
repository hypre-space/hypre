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
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




#ifndef hypre_BLOCKTRIDIAG_HEADER
#define hypre_BLOCKTRIDIAG_HEADER

#include "parcsr_mv/parcsr_mv.h"
#include "parcsr_ls/parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

typedef struct
{
   int    num_sweeps;  
   int    relax_type;   
   int    *index_set1, *index_set2;
   int    print_level;
   double threshold;
   hypre_ParCSRMatrix *A11, *A21, *A22;
   hypre_ParVector    *F1, *U1, *F2, *U2;
   HYPRE_Solver       precon1, precon2;

} hypre_BlockTridiagData;

/*--------------------------------------------------------------------------
 * functions for hypre_BlockTridiag
 *--------------------------------------------------------------------------*/

void *hypre_BlockTridiagCreate();
int  hypre_BlockTridiagDestroy(void *);
int  hypre_BlockTridiagSetup(void * , hypre_ParCSRMatrix *,
                             hypre_ParVector *, hypre_ParVector *);
int  hypre_BlockTridiagSolve(void * , hypre_ParCSRMatrix *,
                             hypre_ParVector *, hypre_ParVector *);
int  hypre_BlockTridiagSetIndexSet(void *, int, int *);
int  hypre_BlockTridiagSetAMGStrengthThreshold(void *, double);
int  hypre_BlockTridiagSetAMGNumSweeps(void *, int);
int  hypre_BlockTridiagSetAMGRelaxType(void *, int);
int  hypre_BlockTridiagSetPrintLevel(void *, int);

#endif

