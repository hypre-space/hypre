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




/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"


int BlockJacobiILUPcKspSolve( void *data, Vec x, Vec b )
     /* Uses ILU as an approximate linear system solver on each
        processor as the block solver in BlockJacobi Preconditioner */
{


  BJData   *bj_data = (BJData *) data;
  SLES     *sles_ptr;  
  int       flg, its;


  sles_ptr = BJDataSles_ptr( bj_data );

  /* Call Petsc solver */
#if 0
  printf("about to call slessolve\n");
#endif

  flg = SLESSolve(*sles_ptr,b,x,&its); CHKERRA(flg);

#if 0
  PetscPrintf(MPI_COMM_WORLD, "iterations = %d\n",its);
#endif


  return flg;
}

