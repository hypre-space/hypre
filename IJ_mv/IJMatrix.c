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
 * hypre_IJMatrix interface
 *
 *****************************************************************************/

#include "./IJ_mv.h"

#include "../HYPRE.h"

/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the row partitioning 

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
hypre_IJMatrixGetRowPartitioning( HYPRE_IJMatrix matrix ,
				  int    **row_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- hypre_IJMatrixGetRowPartitioning\n");
      exit(1);
   }

   if ( hypre_IJMatrixRowPartitioning(ijmatrix))
      *row_partitioning = hypre_IJMatrixRowPartitioning(ijmatrix);
   else
      return -1;

   return -99;
}
/*--------------------------------------------------------------------------
 * hypre_IJMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

/**
Returns a pointer to the column partitioning

@return integer error code
@param IJMatrix [IN]
The ijmatrix to be pointed to.
*/

int
hypre_IJMatrixGetColPartitioning( HYPRE_IJMatrix matrix ,
				  int    **col_partitioning )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      printf("Variable ijmatrix is NULL -- hypre_IJMatrixGetColPartitioning\n");
      exit(1);
   }

   if ( hypre_IJMatrixColPartitioning(ijmatrix))
      *col_partitioning = hypre_IJMatrixColPartitioning(ijmatrix);
   else
      return -1;

   return -99;
}
/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

int 
hypre_IJMatrixSetObject( HYPRE_IJMatrix  matrix, 
                         void           *object )
{
   hypre_IJMatrix *ijmatrix = (hypre_IJMatrix *) matrix;

   if (hypre_IJMatrixObject(ijmatrix) != NULL)
   {
      printf("Referencing a new IJMatrix object can orphan an old -- ");
      printf("hypre_IJMatrixSetObject\n");
      exit(1);
   }

   hypre_IJMatrixObject(ijmatrix) = object;

   return 0;
}
