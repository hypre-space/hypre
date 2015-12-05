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



/******************************************************************************
 *
 * hypre_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"
#include "fortran.h"


/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijmatrixsetobject, HYPRE_IJMATRIXSETOBJECT)(
                                                     long int *matrix,
                                                     long int *object,
                                                     int      *ierr    )
{
   *ierr = (int) ( hypre_IJMatrixSetObject( (HYPRE_IJMatrix) *matrix,
                                            (void *)         *object  ) );
}

