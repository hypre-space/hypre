/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Member functions for hypre_AuxParVector class.
 *
 *****************************************************************************/

#include "IJ_mv.h"
#include "aux_par_vector.h"

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorCreate
 *--------------------------------------------------------------------------*/

int
hypre_AuxParVectorCreate( hypre_AuxParVector **aux_vector)
{
   hypre_AuxParVector  *vector;
   
   vector = hypre_CTAlloc(hypre_AuxParVector, 1);
  
   /* set defaults */
   hypre_AuxParVectorMaxOffProcElmts(vector) = 0;
   hypre_AuxParVectorCurrentNumElmts(vector) = 0;
   /* stash for setting or adding off processor values */
   hypre_AuxParVectorOffProcI(vector) = NULL;
   hypre_AuxParVectorOffProcData(vector) = NULL;


   *aux_vector = vector;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParVectorDestroy( hypre_AuxParVector *vector )
{
   int ierr=0;

   if (vector)
   {
      if (hypre_AuxParVectorOffProcI(vector))
      	    hypre_TFree(hypre_AuxParVectorOffProcI(vector));
      if (hypre_AuxParVectorOffProcData(vector))
      	    hypre_TFree(hypre_AuxParVectorOffProcData(vector));
      hypre_TFree(vector);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParVectorInitialize( hypre_AuxParVector *vector )
{
   int max_off_proc_elmts = hypre_AuxParVectorMaxOffProcElmts(vector);

   /* allocate stash for setting or adding off processor values */
   if (max_off_proc_elmts > 0)
   {
      hypre_AuxParVectorOffProcI(vector) = hypre_CTAlloc(int,
		max_off_proc_elmts);
      hypre_AuxParVectorOffProcData(vector) = hypre_CTAlloc(double,
		max_off_proc_elmts);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

int 
hypre_AuxParVectorSetMaxOffPRocElmts( hypre_AuxParVector *vector,
					    int max_off_proc_elmts )
{
   int ierr = 0;
   hypre_AuxParVectorMaxOffProcElmts(vector) = max_off_proc_elmts;
   return ierr;
}

