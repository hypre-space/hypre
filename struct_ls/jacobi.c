/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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
 * $Revision$
 ***********************************************************************EHEADER*/




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
 * hypre_JacobiCreate
 *--------------------------------------------------------------------------*/

void *
hypre_JacobiCreate( MPI_Comm  comm )
{
   hypre_JacobiData *jacobi_data;
   void              *relax_data;
   hypre_Index       stride;
   hypre_Index       indices[1];

   jacobi_data = hypre_CTAlloc(hypre_JacobiData, 1);
   relax_data = hypre_PointRelaxCreate(comm);
   hypre_PointRelaxSetNumPointsets(relax_data, 1);
   hypre_SetIndex(stride, 1, 1, 1);
   hypre_SetIndex(indices[0], 0, 0, 0);
   hypre_PointRelaxSetPointset(relax_data, 0, 1, stride, indices);
   hypre_PointRelaxSetTol(relax_data,1.0e-6);
   (jacobi_data -> relax_data) = relax_data;

   return (void *) jacobi_data;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiDestroy
 *--------------------------------------------------------------------------*/

int
hypre_JacobiDestroy( void *jacobi_vdata )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   if (jacobi_data)
   {
      hypre_PointRelaxDestroy(jacobi_data -> relax_data);
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
 * hypre_JacobiGetTol
 *--------------------------------------------------------------------------*/

int
hypre_JacobiGetTol( void   *jacobi_vdata,
                    double *tol          )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxGetTol((jacobi_data -> relax_data), tol);

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
 * hypre_JacobiGetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_JacobiGetMaxIter( void  *jacobi_vdata,
                        int  * max_iter     )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxGetMaxIter((jacobi_data -> relax_data),
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
 * hypre_JacobiGetZeroGuess
 *--------------------------------------------------------------------------*/

int
hypre_JacobiGetZeroGuess( void  *jacobi_vdata,
                          int  * zero_guess   )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxGetZeroGuess((jacobi_data -> relax_data),
                                       zero_guess);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_JacobiGetNumIterations
 *--------------------------------------------------------------------------*/

int
hypre_JacobiGetNumIterations( void  *jacobi_vdata,
                              int  * num_iterations   )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   int               ierr = 0;

   ierr = hypre_PointRelaxGetNumIterations((jacobi_data -> relax_data),
                                           num_iterations );

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


/*--------------------------------------------------------------------------
 * hypre_JacobiGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int hypre_JacobiGetFinalRelativeResidualNorm( void * jacobi_vdata, double * norm )
{
   hypre_JacobiData *jacobi_data = jacobi_vdata;
   void *relax_data = jacobi_data -> relax_data;

   return hypre_PointRelaxGetFinalRelativeResidualNorm( relax_data, norm);
}
