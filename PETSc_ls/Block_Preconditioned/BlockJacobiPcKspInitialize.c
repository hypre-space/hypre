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
 * $Revision$
 ***********************************************************************EHEADER*/



/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"


void *BlockJacobiINCFACTPcKspInitialize( void *in_ptr )
     /* Initializes solver structure */
{
   SLES       *sles;
   PC          pc;
   KSP         ksp;

   BJData     *BJ_data;
   int         i, ierr, flg, size, first_row, last_row;



   /* Allocate structure for holding solver data */
   BJ_data = (BJData *) ctalloc( BJData, 1);

   /* Initialize components of BJ_data */
   BJDataA_is_true(BJ_data) = 0;
   BJDataSles_ptr(BJ_data)  = NULL;
   BJDataSlesOwner(BJ_data) = NULL;
   BJDataSystemMatrixPtr(BJ_data)   = NULL;
   BJDataPreconditionerMatrixPtr(BJ_data)   = NULL;
  
   /* Initial setup of incfact_data structure */
   BJDataLsData(BJ_data) = incfact_initialize( (void *) NULL );

   /* Return created BJ structure to calling routine */
   return( BJ_data );

}

int BlockJacobiINCFACTPcKspFinalize (void *data )
{
  BJData      *BJ_data = data;

  incfact_free(BJDataLsData(BJ_data));
  FreeMatrixLimited(BJDataA(BJ_data));
  if( BJDataSlesOwner( BJ_data) == BJLibrary )
  {  
     SLESDestroy(*(BJDataSles_ptr(BJ_data)));
     tfree( BJDataSles_ptr(BJ_data) );
  }
  tfree(BJ_data);

}
