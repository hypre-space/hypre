/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "general.h"
#include "HYPRE_seq_matrix.h"
#include "./CSR_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_NewCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_NewCSRMatrix( )
{
   hypre_CSRMatrix  *matrix;


   matrix = hypre_CTAlloc(hypre_CSRMatrix, 1);

   hypre_CSRMatrixData(matrix)        = NULL;
   hypre_CSRMatrixIA(matrix)        = NULL;
   hypre_CSRMatrixJA(matrix) =        NULL;

   /* set defaults */
   hypre_CSRMatrixSize(matrix) = -1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_FreeCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_FreeCSRMatrix( hypre_CSRMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_Tfree(hypre_CSRMatrixData(matrix));
      hypre_Tfree(hypre_CSRMatrixIA(matrix));
      hypre_Tfree(hypre_CSRMatrixJA(matrix));

      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_LimitedFreeCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_LimitedFreeCSRMatrix( hypre_CSRMatrix *matrix )
{
   int  ierr=0;

   if (matrix)
   {
      hypre_TFree(matrix);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_InitializeCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_InitializeCSRMatrix( hypre_CSRMatrix *matrix )
{
   int    ierr=0;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_AssembleCSRMatrix
 *--------------------------------------------------------------------------*/

int 
hypre_AssembleCSRMatrix( hypre_CSRMatrix *matrix )
{
   int    ierr=0;

   if ( hypre_CSRMatrixN( matrix ) < 0 )
   {
      return(-1);
   }

   if ( hypre_CSRMatrixData( matrix ) == NULL )
   {
      return(-1);
   }

   if ( hypre_CSRMatrixIA( matrix ) == NULL )
   {
      return(-1);
   }

   if ( hypre_CSRMatrixJA( matrix ) == NULL )
   {
      return(-1);
   }

   return(ierr);
}


/*--------------------------------------------------------------------------
 * hypre_PrintCSRMatrix
 *--------------------------------------------------------------------------*/

void
hypre_PrintCSRMatrix(hypre_CSRMatrix *matrix  )
{
   printf("Stub for hypre_PrintCSRMatrix\n");
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/
/*--------------------------------------------------------------------------
 * hypre_SetCSRMatrixData
 *--------------------------------------------------------------------------*/

int 
hypre_SetCSRMatrixData( hypre_CSRMatrix *matrix, double *data )
{
   int    ierr=0;

   hypre_CSRMatrixData(matrix) = data;
   
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetCSRMatrixIA
 *--------------------------------------------------------------------------*/

int 
hypre_SetCSRMatrixIA( hypre_CSRMatrix *matrix, int *ia )
{
   int    ierr=0;

   hypre_CSRMatrixIA(matrix) = ia;
   
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetCSRMatrixJA
 *--------------------------------------------------------------------------*/

int 
hypre_SetCSRMatrixJA( hypre_CSRMatrix *matrix, int *ja )
{
   int    ierr=0;

   hypre_CSRMatrixJA(matrix) = ja;
   
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SetCSRMatrixN
 *--------------------------------------------------------------------------*/

int 
hypre_SetCSRMatrixN( hypre_CSRMatrix *matrix, int n )
{
   int    ierr=0;

   hypre_CSRMatrixSize(matrix) = n;
   
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_GetCSRMatrixData
 *--------------------------------------------------------------------------*/

double *
hypre_GetCSRMatrixData( hypre_CSRMatrix *matrix )
{
   return( hypre_CSRMatrixData(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_GetCSRMatrixIA
 *--------------------------------------------------------------------------*/

int *
hypre_GetCSRMatrixIA( hypre_CSRMatrix *matrix )
{
   return( hypre_CSRMatrixIA(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_GetCSRMatrixJA
 *--------------------------------------------------------------------------*/

int *
hypre_GetCSRMatrixJA( hypre_CSRMatrix *matrix )
{
   return( hypre_CSRMatrixJA(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_GetCSRMatrixN
 *--------------------------------------------------------------------------*/

int 
hypre_GetCSRMatrixN( hypre_CSRMatrix *matrix )
{
   return( hypre_CSRMatrixN(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_GetCSRMatrixNNZ
 *--------------------------------------------------------------------------*/

int 
hypre_GetCSRMatrixNNZ( hypre_CSRMatrix *matrix )
{
   return(hypre_CSRMatrixIA(matrix)[hypre_CSRMatrixSize(matrix)]-1);
}

