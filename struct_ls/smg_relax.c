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
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm             *comm;

   double                tol;
   int                   max_iter;
   int                   zero_guess;
                       
   int                   num_spaces;
   int                  *space_indices;
   int                  *space_strides;
                       
   int                   num_pre_spaces;
   int                  *pre_space_ranks;  /* Ranks of entries in `space_'
                                              arrays above.                */

   zzz_Index            *base_index;
   zzz_Index            *base_stride;

   int                   stencil_dim;
                       
   zzz_StructMatrix     *A;
   zzz_StructVector     *b;
   zzz_StructVector     *x;
   zzz_StructVector     *temp_vec;

   zzz_StructMatrix     *A_sol;       /* Coefficients of A that make up
                                         the (sol)ve part of the relaxation */
   zzz_StructMatrix     *A_rem;       /* Coefficients of A (rem)aining:
                                         A_rem = A - A_sol                  */
   void                **residual_data;  /* Array of size `num_spaces' */
   void                **solve_data;     /* Array of size `num_spaces' */

   /* log info (always logged) */
   int                   num_iterations;

} zzz_SMGRelaxData;

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGRelaxInitialize( MPI_Comm *comm )
{
   zzz_SMGRelaxData *relax_data;

   relax_data = zzz_CTAlloc(zzz_SMGRelaxData, 1);

   (relax_data -> comm) = comm;
   (relax_data -> base_index)  = zzz_NewIndex();
   (relax_data -> base_stride) = zzz_NewIndex();

   /* set defaults */
   (relax_data -> tol)        = 1.0e-06;
   (relax_data -> max_iter)   = 1000;
   (relax_data -> zero_guess) = 0;
   (relax_data -> num_spaces) = 1;
   (relax_data -> space_indices) = zzz_TAlloc(int, 1);
   (relax_data -> space_strides) = zzz_TAlloc(int, 1);
   (relax_data -> space_indices[0]) = 0;
   (relax_data -> space_strides[0]) = 1;
   (relax_data -> num_pre_spaces) = 0;
   (relax_data -> pre_space_ranks) = NULL;
   zzz_SetIndex((relax_data -> base_index), 0, 0, 0);
   zzz_SetIndex((relax_data -> base_stride), 1, 1, 1);

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetup( void             *relax_vdata,
                   zzz_StructMatrix *A,
                   zzz_StructVector *b,
                   zzz_StructVector *x,
                   zzz_StructVector *temp_vec    )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;

   int                   num_spaces    = (relax_data -> num_spaces);
   int                  *space_indices = (relax_data -> space_indices);
   int                  *space_strides = (relax_data -> space_strides);

   zzz_StructStencil    *stencil       = zzz_StructMatrixStencil(A);     
   zzz_Index           **stencil_shape = zzz_StructStencilShape(stencil);
   int                   stencil_size  = zzz_StructStencilSize(stencil); 
   int                   stencil_dim   = zzz_StructStencilDim(stencil);
                       
   zzz_StructMatrix     *A_sol;
   zzz_StructMatrix     *A_rem;
   void                **residual_data;
   void                **solve_data;

   zzz_Index            *base_index;
   zzz_Index            *base_stride;

   int                   num_stencil_indices;
   int                  *stencil_indices;
                       
   int                   i;

   int                   ierr;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   if ((relax_data -> A) != NULL)
   {
      zzz_FreeStructMatrixMask(relax_data -> A_sol);
      zzz_FreeStructMatrixMask(relax_data -> A_rem);
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         zzz_SMGResidualFinalize(relax_data -> residual_data[i]);
         if (stencil_dim > 2)
            zzz_SMGFinalize(relax_data -> solve_data[i]);
         else
            zzz_CyclicReductionFinalize(relax_data -> solve_data[i]);
      }
      zzz_TFree(relax_data -> residual_data);
      zzz_TFree(relax_data -> solve_data);
   }

   /*----------------------------------------------------------
    * Set up data for relaxations
    *----------------------------------------------------------*/

   (relax_data -> stencil_dim) = stencil_dim;
   (relax_data -> A)           = A;
   (relax_data -> b)           = b;
   (relax_data -> x)           = x;
   (relax_data -> temp_vec)    = temp_vec;

   base_index  = zzz_NewIndex();
   base_stride = zzz_NewIndex();
   zzz_CopyIndex((relax_data -> base_index),  base_index);
   zzz_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = zzz_TAlloc(int, stencil_size);

   /* set up A_sol matrix */
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (zzz_IndexD(stencil_shape[i], (stencil_dim - 1)) == 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A_sol = zzz_NewStructMatrixMask(A, num_stencil_indices, stencil_indices);
   zzz_StructStencilDim(zzz_StructMatrixStencil(A_sol)) = stencil_dim - 1;

   /* set up A_rem matrix */
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (zzz_IndexD(stencil_shape[i], (stencil_dim - 1)) != 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A_rem = zzz_NewStructMatrixMask(A, num_stencil_indices, stencil_indices);

   /* Set up residual_data and solve_data */
   residual_data = zzz_TAlloc(void *, num_spaces);
   solve_data    = zzz_TAlloc(void *, num_spaces);
   for (i = 0; i < num_spaces; i++)
   {
      zzz_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      zzz_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      residual_data[i] = zzz_SMGResidualInitialize();
      zzz_SMGResidualSetBase(residual_data[i], base_index, base_stride);
      zzz_SMGResidualSetup(residual_data[i], A_rem, x, b, temp_vec);

      if (stencil_dim > 2)
      {
         solve_data[i] = zzz_SMGInitialize(relax_data -> comm);
         zzz_SMGSetBase(solve_data[i], base_index, base_stride);
         zzz_SMGSetTol(solve_data[i], 0.0);
         zzz_SMGSetMaxIter(solve_data[i], 1);
         zzz_SMGSetZeroGuess(solve_data[i]);
         zzz_SMGSetup(solve_data[i], A_sol, temp_vec, x);
      }
      else
      {
         solve_data[i] = zzz_CyclicReductionInitialize(relax_data -> comm);
         zzz_CyclicReductionSetBase(solve_data[i], base_index, base_stride);
         zzz_CyclicReductionSetup(solve_data[i], A_sol, temp_vec, x);
      }
   }

   zzz_TFree(stencil_indices);

   zzz_FreeIndex(base_index);
   zzz_FreeIndex(base_stride);

   (relax_data -> A_sol)         = A_sol;
   (relax_data -> A_rem)         = A_rem;
   (relax_data -> residual_data) = residual_data;
   (relax_data -> solve_data)    = solve_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelax
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelax( void             *relax_vdata,
              zzz_StructVector *b,
              zzz_StructVector *x               )
{
   zzz_SMGRelaxData   *relax_data = relax_vdata;

   int                 max_iter        = (relax_data -> max_iter);
   int                 zero_guess      = (relax_data -> zero_guess);
   int                 num_spaces      = (relax_data -> num_spaces);
   int                 num_pre_spaces  = (relax_data -> num_pre_spaces);
   int                *pre_space_ranks = (relax_data -> pre_space_ranks);
   int                 stencil_dim     = (relax_data -> stencil_dim);
   zzz_StructVector   *temp_vec        = (relax_data -> temp_vec);
   void              **residual_data   = (relax_data -> residual_data);
   void              **solve_data      = (relax_data -> solve_data);

   int                 i, is;

   int                 ierr;

   /*----------------------------------------------------------
    * Zero out initial guess if necessary
    *----------------------------------------------------------*/

   if (zero_guess)
   {
      zzz_SetStructVectorConstantValues(x, 0.0);
   }

   /*----------------------------------------------------------
    * Do pre-relaxation
    *----------------------------------------------------------*/

   for (i = 0; i < num_pre_spaces; i++)
   {
      is = pre_space_ranks[i];

      zzz_SMGResidual(residual_data[is], x, b, temp_vec);

      if (stencil_dim > 2)
         zzz_SMGSolve(solve_data[is], temp_vec, x);
      else
         zzz_CyclicReduction(solve_data[is], temp_vec, x);
   }

   /*----------------------------------------------------------
    * Do regular relaxation iterations
    *----------------------------------------------------------*/

   for (i = 0; i < max_iter; i++)
   {
      for (is = 0; is < num_spaces; is++)
      {
         zzz_SMGResidual(residual_data[is], x, b, temp_vec);

         if (stencil_dim > 2)
            zzz_SMGSolve(solve_data[is], temp_vec, x);
         else
            zzz_CyclicReduction(solve_data[is], temp_vec, x);
      }

      (relax_data -> num_iterations) = (i + 1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetTol
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetTol( void   *relax_vdata,
                    double  tol             )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetMaxIter
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetMaxIter( void *relax_vdata,
                        int   max_iter        )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> max_iter) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetZeroGuess
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetZeroGuess( void *relax_vdata )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> zero_guess) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetNumSpaces
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetNumSpaces( void *relax_vdata,
                          int   num_spaces      )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               i;
   int               ierr = 0;

   (relax_data -> num_spaces) = num_spaces;

   zzz_TFree(relax_data -> space_indices);
   zzz_TFree(relax_data -> space_strides);
   (relax_data -> space_indices) = zzz_TAlloc(int, num_spaces);
   (relax_data -> space_strides) = zzz_TAlloc(int, num_spaces);

   for (i = 0; i < num_spaces; i++)
   {
      (relax_data -> space_indices[i]) = 0;
      (relax_data -> space_strides[i]) = 1;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetNumPreSpaces
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetNumPreSpaces( void *relax_vdata,
                             int   num_pre_spaces      )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               i;
   int               ierr = 0;

   (relax_data -> num_pre_spaces) = num_pre_spaces;

   zzz_TFree(relax_data -> pre_space_ranks);
   (relax_data -> pre_space_ranks) = zzz_TAlloc(int, num_pre_spaces);

   for (i = 0; i < num_pre_spaces; i++)
      (relax_data -> pre_space_ranks[i]) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetSpace
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetSpace( void *relax_vdata,
                      int   i,
                      int   space_index,
                      int   space_stride    )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> space_indices[i]) = space_index;
   (relax_data -> space_strides[i]) = space_stride;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetPreSpace
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetPreSpace( void *relax_vdata,
                         int   i,
                         int   pre_space_rank  )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> pre_space_ranks[i]) = pre_space_rank;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetBase
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGRelaxSetBase( void      *relax_vdata,
                     zzz_Index *base_index,
                     zzz_Index *base_stride )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int          d;
   int          ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD((relax_data -> base_index),  d) =
         zzz_IndexD(base_index,  d);
      zzz_IndexD((relax_data -> base_stride), d) =
         zzz_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxFinalize( void *relax_vdata )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;

   int               stencil_dim = (relax_data -> stencil_dim);
   int               is;
   int               ierr;

   if (relax_data)
   {
      zzz_TFree(relax_data -> space_indices);
      zzz_TFree(relax_data -> space_strides);
      zzz_TFree(relax_data -> pre_space_ranks);
      zzz_FreeIndex(relax_data -> base_index);
      zzz_FreeIndex(relax_data -> base_stride);
      zzz_FreeStructMatrixMask(relax_data -> A_sol);
      zzz_FreeStructMatrixMask(relax_data -> A_rem);
      for (is = 0; is < (relax_data -> num_spaces); is++)
      {
         zzz_SMGResidualFinalize(relax_data -> residual_data[is]);
         if (stencil_dim > 2)
            zzz_SMGFinalize(relax_data -> solve_data[is]);
         else
            zzz_CyclicReductionFinalize(relax_data -> solve_data[is]);
      }
      zzz_TFree(relax_data -> residual_data);
      zzz_TFree(relax_data -> solve_data);
      zzz_TFree(relax_data);
   }

   return ierr;
}

