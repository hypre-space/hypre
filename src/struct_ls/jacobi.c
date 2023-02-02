/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

typedef struct
{
   void  *relax_data;

} hypre_JacobiData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_JacobiCreate( MPI_Comm  comm )
{
   hypre_JacobiData *jacobi_data;
   void              *relax_data;
   hypre_Index       stride;
   hypre_Index       indices[1];

   jacobi_data = hypre_CTAlloc(hypre_JacobiData,  1, HYPRE_MEMORY_HOST);
   relax_data = hypre_PointRelaxCreate(comm);
   hypre_PointRelaxSetNumPointsets(relax_data, 1);
   hypre_SetIndex3(stride, 1, 1, 1);
   hypre_SetIndex3(indices[0], 0, 0, 0);
   hypre_PointRelaxSetPointset(relax_data, 0, 1, stride, indices);
   hypre_PointRelaxSetTol(relax_data, 1.0e-6);
   (jacobi_data -> relax_data) = relax_data;

   return (void *) jacobi_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiDestroy( void *jacobi_vdata )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   if (jacobi_data)
   {
      hypre_PointRelaxDestroy(jacobi_data -> relax_data);
      hypre_TFree(jacobi_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiSetup( void               *jacobi_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxSetup((jacobi_data -> relax_data), A, b, x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiSolve( void               *jacobi_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *b,
                   hypre_StructVector *x            )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelax((jacobi_data -> relax_data), A, b, x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiSetTol( void   *jacobi_vdata,
                    HYPRE_Real  tol          )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxSetTol((jacobi_data -> relax_data), tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiGetTol( void   *jacobi_vdata,
                    HYPRE_Real *tol          )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxGetTol((jacobi_data -> relax_data), tol);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiSetMaxIter( void  *jacobi_vdata,
                        HYPRE_Int    max_iter     )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxSetMaxIter((jacobi_data -> relax_data), max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiGetMaxIter( void  *jacobi_vdata,
                        HYPRE_Int  * max_iter     )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxGetMaxIter((jacobi_data -> relax_data), max_iter);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiSetZeroGuess( void  *jacobi_vdata,
                          HYPRE_Int    zero_guess   )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxSetZeroGuess((jacobi_data -> relax_data), zero_guess);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiGetZeroGuess( void  *jacobi_vdata,
                          HYPRE_Int  * zero_guess   )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxGetZeroGuess((jacobi_data -> relax_data), zero_guess);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiGetNumIterations( void  *jacobi_vdata,
                              HYPRE_Int  * num_iterations   )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxGetNumIterations((jacobi_data -> relax_data), num_iterations );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_JacobiSetTempVec( void               *jacobi_vdata,
                        hypre_StructVector *t            )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;

   hypre_PointRelaxSetTempVec((jacobi_data -> relax_data), t);

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_JacobiGetFinalRelativeResidualNorm( void * jacobi_vdata,
                                                    HYPRE_Real * norm )
{
   hypre_JacobiData *jacobi_data = (hypre_JacobiData *)jacobi_vdata;
   void *relax_data = jacobi_data -> relax_data;

   return hypre_PointRelaxGetFinalRelativeResidualNorm( relax_data, norm );
}
