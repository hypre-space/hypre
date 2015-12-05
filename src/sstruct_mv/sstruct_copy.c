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
 * SStruct copy routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructPCopy
 *--------------------------------------------------------------------------*/

int
hypre_SStructPCopy( hypre_SStructPVector *px,
                    hypre_SStructPVector *py )
{
   int ierr = 0;
   int nvars = hypre_SStructPVectorNVars(px);
   int var;

   for (var = 0; var < nvars; var++)
   {
      hypre_StructCopy(hypre_SStructPVectorSVector(px, var),
                       hypre_SStructPVectorSVector(py, var));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPartialPCopy: Copy the components on only a subset of the
 * pgrid. For each box of an sgrid, an array of subboxes are copied.
 *--------------------------------------------------------------------------*/

int
hypre_SStructPartialPCopy( hypre_SStructPVector *px,
                           hypre_SStructPVector *py,
                           hypre_BoxArrayArray **array_boxes )
{
   int ierr = 0;
   int nvars = hypre_SStructPVectorNVars(px);
   hypre_BoxArrayArray  *boxes;
   int var;

   for (var = 0; var < nvars; var++)
   {
      boxes= array_boxes[var];
      hypre_StructPartialCopy(hypre_SStructPVectorSVector(px, var),
                              hypre_SStructPVectorSVector(py, var),
                              boxes);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructCopy
 *--------------------------------------------------------------------------*/

int
hypre_SStructCopy( hypre_SStructVector *x,
                   hypre_SStructVector *y )
{
   int ierr = 0;

   int nparts = hypre_SStructVectorNParts(x);
   int part;

   int x_object_type= hypre_SStructVectorObjectType(x);
   int y_object_type= hypre_SStructVectorObjectType(y);

   if (x_object_type != y_object_type)
   {
       printf("vector object types different- cannot perform SStructCopy\n");
       return ierr;
   }


   if (x_object_type == HYPRE_SSTRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPCopy(hypre_SStructVectorPVector(x, part),
                            hypre_SStructVectorPVector(y, part));
      }
   }

   else if (x_object_type == HYPRE_PARCSR)
   {
      hypre_ParVector  *x_par;
      hypre_ParVector  *y_par;

      hypre_SStructVectorConvert(x, &x_par);
      hypre_SStructVectorConvert(y, &y_par);

      hypre_ParVectorCopy(x_par, y_par);
   }

   return ierr;
}
