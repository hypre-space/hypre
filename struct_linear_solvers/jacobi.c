/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_JacobiData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   void  *relax_data;

} hypre_JacobiData;

/*--------------------------------------------------------------------------
 * hypre_JacobiInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_JacobiInitialize( MPI_Comm  comm )
{
   hypre_JacobiData *jacobi_data;
   void              *relax_data;
   hypre_Index       stride;
   hypre_Index       indices[1];

   jacobi_data = hypre_CTAlloc(hypre_JacobiData, 1);
   relax_data = hypre_PointRelaxInitialize(comm);
   hypre_PointRelaxSetNumPointsets(relax_data, 1);
   hypre_SetIndex(stride, 1, 1, 1);
   hypre_SetIndex(indices[0], 0, 0, 0);
   hypre_PointRelaxSetPointset(relax_data, 0, 1, stride, indices);
   (jacobi_data -> relax_data) = relax_data;

   return (void *) jacobi_data;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiFinalize
 *--------------------------------------------------------------------------*/

int
hypre_JacobiFinalize( void *jacobi_vdata )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   if (jacobi_data)
   {
      hypre_PointRelaxFinalize(jacobi_data -> relax_data);
      hypre_TFree(jacobi_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiSetup
 *--------------------------------------------------------------------------*/

int
hypre_JacobiSetup( void               *jacobi_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxSetup((jacobi_data -> relax_data), A, b, x);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiSolve
 *--------------------------------------------------------------------------*/

int
hypre_JacobiSolve( void               *jacobi_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelax((jacobi_data -> relax_data), A, b, x);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiSetTol
 *--------------------------------------------------------------------------*/

int
hypre_JacobiSetTol( void   *jacobi_vdata,
                    double  tol          )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxSetTol((jacobi_data -> relax_data), tol);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_JacobiSetMaxIter( void  *jacobi_vdata,
                        int    max_iter     )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxSetMaxIter((jacobi_data -> relax_data),
                                     max_iter);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiSetZeroGuess
 *--------------------------------------------------------------------------*/

int
hypre_JacobiSetZeroGuess( void  *jacobi_vdata,
                          int    zero_guess   )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxSetZeroGuess((jacobi_data -> relax_data),
                                       zero_guess);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiSetTempVec
 *--------------------------------------------------------------------------*/

int
hypre_JacobiSetTempVec( void               *jacobi_vdata,
                        hypre_StructVector *t            )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxSetTempVec((jacobi_data -> relax_data), t);

   return ierr;
}

