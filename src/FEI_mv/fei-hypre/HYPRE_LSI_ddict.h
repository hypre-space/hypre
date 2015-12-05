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
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * HYPRE_DDICT interface
 *
 *****************************************************************************/

#ifndef __HYPRE_DDICT__
#define __HYPRE_DDICT__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif
extern int HYPRE_LSI_DDICTCreate( MPI_Comm comm, HYPRE_Solver *solver );
extern int HYPRE_LSI_DDICTDestroy( HYPRE_Solver solver );
extern int HYPRE_LSI_DDICTSetFillin( HYPRE_Solver solver, double fillin);
extern int HYPRE_LSI_DDICTSetOutputLevel( HYPRE_Solver solver, int level);
extern int HYPRE_LSI_DDICTSetDropTolerance( HYPRE_Solver solver, double thresh);
extern int HYPRE_LSI_DDICTSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                 HYPRE_ParVector b,   HYPRE_ParVector x );
extern int HYPRE_LSI_DDICTSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                 HYPRE_ParVector b,   HYPRE_ParVector x );
#ifdef __cplusplus
}
#endif

#endif

