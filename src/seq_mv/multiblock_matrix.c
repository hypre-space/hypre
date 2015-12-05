/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Member functions for hypre_MultiblockMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_MultiblockMatrix *
hypre_MultiblockMatrixCreate( )
{
   hypre_MultiblockMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_MultiblockMatrix, 1);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MultiblockMatrixDestroy( hypre_MultiblockMatrix *matrix )
{
   HYPRE_Int  ierr=0, i;

   if (matrix)
   {
      for(i=0; i < hypre_MultiblockMatrixNumSubmatrices(matrix); i++)
         hypre_TFree(hypre_MultiblockMatrixSubmatrix(matrix,i));
      hypre_TFree(hypre_MultiblockMatrixSubmatrices(matrix));
      hypre_TFree(hypre_MultiblockMatrixSubmatrixTypes(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MultiblockMatrixLimitedDestroy( hypre_MultiblockMatrix *matrix )
{
   HYPRE_Int  ierr=0;

   if (matrix)
   {
      hypre_TFree(hypre_MultiblockMatrixSubmatrices(matrix));
      hypre_TFree(hypre_MultiblockMatrixSubmatrixTypes(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MultiblockMatrixInitialize( hypre_MultiblockMatrix *matrix )
{
   HYPRE_Int    ierr=0;

   if( hypre_MultiblockMatrixNumSubmatrices(matrix) <= 0 )
      return(-1);

   hypre_MultiblockMatrixSubmatrixTypes(matrix) = 
      hypre_CTAlloc( HYPRE_Int, hypre_MultiblockMatrixNumSubmatrices(matrix) );

   hypre_MultiblockMatrixSubmatrices(matrix) = 
      hypre_CTAlloc( void *, hypre_MultiblockMatrixNumSubmatrices(matrix) );

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_MultiblockMatrixAssemble( hypre_MultiblockMatrix *matrix )
{
   HYPRE_Int    ierr=0;

   return(ierr);
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_MultiblockMatrixPrint(hypre_MultiblockMatrix *matrix  )
{
   hypre_printf("Stub for hypre_MultiblockMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetNumSubmatrices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MultiblockMatrixSetNumSubmatrices(hypre_MultiblockMatrix *matrix, HYPRE_Int n  )
{
   HYPRE_Int ierr = 0;

   hypre_MultiblockMatrixNumSubmatrices(matrix) = n;
   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MultiblockMatrixSetSubmatrixType(hypre_MultiblockMatrix *matrix, 
                                     HYPRE_Int j,
                                     HYPRE_Int type  )
{
   HYPRE_Int ierr = 0;

   if ( (j<0) || 
         (j >= hypre_MultiblockMatrixNumSubmatrices(matrix)) )
      return(-1);

   hypre_MultiblockMatrixSubmatrixType(matrix,j) = type;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetSubmatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MultiblockMatrixSetSubmatrix(hypre_MultiblockMatrix *matrix, 
                                     HYPRE_Int j,
                                     void *submatrix  )
{
   HYPRE_Int ierr = 0;

   if ( (j<0) || 
         (j >= hypre_MultiblockMatrixNumSubmatrices(matrix)) )
      return(-1);

   hypre_MultiblockMatrixSubmatrix(matrix,j) = submatrix;

   return( ierr );
}


