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


void *BlockJacobiICPcKspInitialize( void *in_ptr )
     /* Initializes solver structure */
{
   SLES       *sles;
   PC          pc;
   KSP         ksp;

   BJData     *BJ_data;
   int         i, ierr, flg, size, first_row, last_row;



   /* Allocate structure for holding solver data */
   BJ_data = (BJData *) ctalloc( BJData, 1);

   /* Create SLES context and set operators */
   sles = (SLES *) ctalloc( SLES, 1);
   ierr = SLESCreate(MPI_COMM_WORLD,sles); CHKERRA(ierr);
   BJDataSles_ptr(BJ_data) = sles;

   /* Set KSP to be GMRES */
   ierr = SLESGetKSP(*sles,&ksp); CHKERRA(ierr);
   ierr = KSPSetType(ksp,KSPGMRES); CHKERRA(ierr);

   /* Set Petsc KSP error tolerance */
   ierr = KSPSetTolerances(ksp,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT,
             PETSC_DEFAULT); CHKERRA(ierr);

#ifdef LocalSolverILU
   /* Default to preconditioning on the right side */
   ierr = KSPSetPreconditionerSide( ksp, PC_RIGHT );
#endif
#ifdef LocalSolverIC
   /* Default to preconditioning on the left side */
   ierr = KSPSetPreconditionerSide( ksp, PC_LEFT );
#endif

   /* Set preconditioner to Additive Schwarz, one block per processor. */
   ierr = SLESGetPC(*sles,&pc); CHKERRA(ierr);
   ierr = PCSetType(pc,PCASM); CHKERRA(ierr);


   /* User can change the above on the command line; import those changes */
   ierr = SLESSetFromOptions( *sles );


   /* From this point on, we override user choices */

   /* Tell Petsc that we are using a nonzero initial guess */
   ierr = KSPSetInitialGuessNonzero( ksp ); CHKERRA(ierr);

   /* Haven't quite got this call correct... -AC
   ierr = PCBJacobiSetLocalBlocks(pc, 1, PETSC_NULL, PETSC_NULL ); CHKERRA(ierr);
   */

   /* Return created BJ structure to calling routine */
   return( BJ_data );

}

int BlockJacobiICPcKspFinalize (void *data )
{
  BJData      *BJ_data = data;

  SLESDestroy(*(BJDataSles_ptr(BJ_data)));
  FreeMatrix(BJDataA(BJ_data));
  ic_free(BJDataLsData(BJ_data));
  tfree(BJ_data);

}
