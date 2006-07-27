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
 * Member functions for hypre_MappedMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_MappedMatrix *
hypre_MappedMatrixCreate(  )
{
   hypre_MappedMatrix  *matrix;


   matrix = hypre_CTAlloc(hypre_MappedMatrix, 1);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_MappedMatrixDestroy( hypre_MappedMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(hypre_MappedMatrixMatrix(matrix));
      hypre_TFree(hypre_MappedMatrixMapData(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_MappedMatrixLimitedDestroy( hypre_MappedMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_MappedMatrixInitialize( hypre_MappedMatrix *matrix )
{
   int    ierr=0;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_MappedMatrixAssemble( hypre_MappedMatrix *matrix )
{
   int    ierr=0;

   if( matrix == NULL )
      return ( -1 ) ;

   if( hypre_MappedMatrixMatrix(matrix) == NULL )
      return ( -1 ) ;

   if( hypre_MappedMatrixColMap(matrix) == NULL )
      return ( -1 ) ;

   if( hypre_MappedMatrixMapData(matrix) == NULL )
      return ( -1 ) ;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_MappedMatrixPrint(hypre_MappedMatrix *matrix  )
{
   printf("Stub for hypre_MappedMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetColIndex
 *--------------------------------------------------------------------------*/

int
hypre_MappedMatrixGetColIndex(hypre_MappedMatrix *matrix, int j  )
{
   return( hypre_MappedMatrixColIndex(matrix,j) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetMatrix
 *--------------------------------------------------------------------------*/

void *
hypre_MappedMatrixGetMatrix(hypre_MappedMatrix *matrix )
{
   return( hypre_MappedMatrixMatrix(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMatrix
 *--------------------------------------------------------------------------*/

int
hypre_MappedMatrixSetMatrix(hypre_MappedMatrix *matrix, void *matrix_data  )
{
   int ierr=0;

   hypre_MappedMatrixMatrix(matrix) = matrix_data;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

int
hypre_MappedMatrixSetColMap(hypre_MappedMatrix *matrix, 
                          int (*ColMap)(int, void *)  )
{
   int ierr=0;

   hypre_MappedMatrixColMap(matrix) = ColMap;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

int
hypre_MappedMatrixSetMapData(hypre_MappedMatrix *matrix, 
                          void *map_data )
{
   int ierr=0;

   hypre_MappedMatrixMapData(matrix) = map_data;

   return(ierr);
}

