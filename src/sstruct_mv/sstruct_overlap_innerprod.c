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
 * $Revision: 2.3 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * SStruct inner product routine for overlapped grids.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPOverlapInnerProd
 *--------------------------------------------------------------------------*/

int
hypre_SStructPOverlapInnerProd( hypre_SStructPVector *px,
                                hypre_SStructPVector *py,
                                double               *presult_ptr )
{
   int    ierr = 0;
   int    nvars = hypre_SStructPVectorNVars(px);
   double presult;
   double sresult;
   int    var;

   presult = 0.0;
   for (var = 0; var < nvars; var++)
   {
      sresult = hypre_StructOverlapInnerProd(hypre_SStructPVectorSVector(px, var),
                                             hypre_SStructPVectorSVector(py, var));
      presult += sresult;
   }

   *presult_ptr = presult;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructOverlapInnerProd
 *--------------------------------------------------------------------------*/

int
hypre_SStructOverlapInnerProd( hypre_SStructVector *x,
                               hypre_SStructVector *y,
                               double              *result_ptr )
{
   int    ierr = 0;
   int    nparts = hypre_SStructVectorNParts(x);
   double result;
   double presult;
   int    part;

   result = 0.0;
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPOverlapInnerProd(hypre_SStructVectorPVector(x, part),
                                     hypre_SStructVectorPVector(y, part), &presult);
      result += presult;
   }

   *result_ptr = result;

   return ierr;
}
