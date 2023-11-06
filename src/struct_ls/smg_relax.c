/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int               setup_temp_vec;
   HYPRE_Int               setup_a_rem;
   HYPRE_Int               setup_a_sol;

   MPI_Comm                comm;

   HYPRE_Int               memory_use;
   HYPRE_Real              tol;
   HYPRE_Int               max_iter;
   HYPRE_Int               zero_guess;

   HYPRE_Int               num_spaces;
   HYPRE_Int              *space_indices;
   HYPRE_Int              *space_strides;

   HYPRE_Int               num_pre_spaces;
   HYPRE_Int               num_reg_spaces;
   HYPRE_Int              *pre_space_ranks;
   HYPRE_Int              *reg_space_ranks;

   hypre_Index             base_index;
   hypre_Index             base_stride;
   hypre_BoxArray         *base_box_array;

   HYPRE_Int               stencil_dim;

   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;

   hypre_StructVector     *temp_vec;
   hypre_StructMatrix     *A_sol;  /* Coefficients of A that make up
                                      the (sol)ve part of the relaxation */
   hypre_StructMatrix     *A_rem;  /* Coefficients of A (rem)aining:
                                      A_rem = A - A_sol                  */
   void                  **residual_data;  /* Array of size `num_spaces' */
   void                  **solve_data;     /* Array of size `num_spaces' */

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;

   HYPRE_Int               num_pre_relax;
   HYPRE_Int               num_post_relax;

   HYPRE_Int               max_level;
} hypre_SMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SMGRelaxCreate( MPI_Comm  comm )
{
   hypre_SMGRelaxData *relax_data;

   relax_data = hypre_CTAlloc(hypre_SMGRelaxData,  1, HYPRE_MEMORY_HOST);
   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;
   (relax_data -> comm)           = comm;
   (relax_data -> base_box_array) = NULL;
   (relax_data -> time_index)     = hypre_InitializeTiming("SMGRelax");
   /* set defaults */
   (relax_data -> memory_use)         = 0;
   (relax_data -> tol)                = 1.0e-06;
   (relax_data -> max_iter)           = 1000;
   (relax_data -> zero_guess)         = 0;
   (relax_data -> num_spaces)         = 1;
   (relax_data -> space_indices)      = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
   (relax_data -> space_strides)      = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
   (relax_data -> space_indices[0])   = 0;
   (relax_data -> space_strides[0])   = 1;
   (relax_data -> num_pre_spaces)     = 0;
   (relax_data -> num_reg_spaces)     = 1;
   (relax_data -> pre_space_ranks)    = NULL;
   (relax_data -> reg_space_ranks)    = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
   (relax_data -> reg_space_ranks[0]) = 0;
   hypre_SetIndex3((relax_data -> base_index), 0, 0, 0);
   hypre_SetIndex3((relax_data -> base_stride), 1, 1, 1);
   (relax_data -> A)                  = NULL;
   (relax_data -> b)                  = NULL;
   (relax_data -> x)                  = NULL;
   (relax_data -> temp_vec)           = NULL;

   (relax_data -> num_pre_relax)  = 1;
   (relax_data -> num_post_relax) = 1;
   (relax_data -> max_level)      = -1;
   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxDestroyTempVec( void *relax_vdata )
{
   hypre_SMGRelaxData  *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   hypre_StructVectorDestroy(relax_data -> temp_vec);
   (relax_data -> setup_temp_vec) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxDestroyARem( void *relax_vdata )
{
   hypre_SMGRelaxData  *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   HYPRE_Int            i;

   if (relax_data -> A_rem)
   {
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         hypre_SMGResidualDestroy(relax_data -> residual_data[i]);
      }
      hypre_TFree(relax_data -> residual_data, HYPRE_MEMORY_HOST);
      hypre_StructMatrixDestroy(relax_data -> A_rem);
      (relax_data -> A_rem) = NULL;
   }
   (relax_data -> setup_a_rem) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxDestroyASol( void *relax_vdata )
{
   hypre_SMGRelaxData  *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   HYPRE_Int            stencil_dim;
   HYPRE_Int            i;

   if (relax_data -> A_sol)
   {
      stencil_dim = (relax_data -> stencil_dim);
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         if (stencil_dim > 2)
         {
            hypre_SMGDestroy(relax_data -> solve_data[i]);
         }
         else
         {
            hypre_CyclicReductionDestroy(relax_data -> solve_data[i]);
         }
      }
      hypre_TFree(relax_data -> solve_data, HYPRE_MEMORY_HOST);
      hypre_StructMatrixDestroy(relax_data -> A_sol);
      (relax_data -> A_sol) = NULL;
   }
   (relax_data -> setup_a_sol) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxDestroy( void *relax_vdata )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   if (relax_data)
   {
      hypre_TFree(relax_data -> space_indices, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> space_strides, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> pre_space_ranks, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> reg_space_ranks, HYPRE_MEMORY_HOST);
      hypre_BoxArrayDestroy(relax_data -> base_box_array);

      hypre_StructMatrixDestroy(relax_data -> A);
      hypre_StructVectorDestroy(relax_data -> b);
      hypre_StructVectorDestroy(relax_data -> x);

      hypre_SMGRelaxDestroyTempVec(relax_vdata);
      hypre_SMGRelaxDestroyARem(relax_vdata);
      hypre_SMGRelaxDestroyASol(relax_vdata);

      hypre_FinalizeTiming(relax_data -> time_index);
      hypre_TFree(relax_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelax( void               *relax_vdata,
                hypre_StructMatrix *A,
                hypre_StructVector *b,
                hypre_StructVector *x           )
{
   hypre_SMGRelaxData   *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   HYPRE_Int             zero_guess;
   HYPRE_Int             stencil_dim;
   hypre_StructVector   *temp_vec;
   hypre_StructMatrix   *A_sol;
   hypre_StructMatrix   *A_rem;
   void                **residual_data;
   void                **solve_data;

   hypre_IndexRef        base_stride;
   hypre_BoxArray       *base_box_a;
   HYPRE_Real            zero = 0.0;

   HYPRE_Int             max_iter;
   HYPRE_Int             num_spaces;
   HYPRE_Int            *space_ranks;

   HYPRE_Int             i, j, k, is;

   /*----------------------------------------------------------
    * Note: The zero_guess stuff is not handled correctly
    * for general relaxation parameters.  It is correct when
    * the spaces are independent sets in the direction of
    * relaxation.
    *----------------------------------------------------------*/

   hypre_BeginTiming(relax_data -> time_index);

   /*----------------------------------------------------------
    * Set up the solver
    *----------------------------------------------------------*/

   /* insure that the solver memory gets fully set up */
   if ((relax_data -> setup_a_sol) > 0)
   {
      (relax_data -> setup_a_sol) = 2;
   }

   hypre_SMGRelaxSetup(relax_vdata, A, b, x);

   zero_guess      = (relax_data -> zero_guess);
   stencil_dim     = (relax_data -> stencil_dim);
   temp_vec        = (relax_data -> temp_vec);
   A_sol           = (relax_data -> A_sol);
   A_rem           = (relax_data -> A_rem);
   residual_data   = (relax_data -> residual_data);
   solve_data      = (relax_data -> solve_data);

   /*----------------------------------------------------------
    * Set zero values
    *----------------------------------------------------------*/

   if (zero_guess)
   {
      base_stride = (relax_data -> base_stride);
      base_box_a = (relax_data -> base_box_array);
      hypre_SMGSetStructVectorConstantValues(x, zero, base_box_a, base_stride);
   }

   /*----------------------------------------------------------
    * Iterate
    *----------------------------------------------------------*/

   for (k = 0; k < 2; k++)
   {
      switch (k)
      {
         /* Do pre-relaxation iterations */
         case 0:
            max_iter    = 1;
            num_spaces  = (relax_data -> num_pre_spaces);
            space_ranks = (relax_data -> pre_space_ranks);
            break;

         /* Do regular relaxation iterations */
         case 1:
            max_iter    = (relax_data -> max_iter);
            num_spaces  = (relax_data -> num_reg_spaces);
            space_ranks = (relax_data -> reg_space_ranks);
            break;
      }

      for (i = 0; i < max_iter; i++)
      {
         for (j = 0; j < num_spaces; j++)
         {
            is = space_ranks[j];

            hypre_SMGResidual(residual_data[is], A_rem, x, b, temp_vec);

            if (stencil_dim > 2)
            {
               hypre_SMGSolve(solve_data[is], A_sol, temp_vec, x);
            }
            else
            {
               hypre_CyclicReduction(solve_data[is], A_sol, temp_vec, x);
            }
         }

         (relax_data -> num_iterations) = (i + 1);
      }
   }

   /*----------------------------------------------------------
    * Free up memory according to memory_use parameter
    *----------------------------------------------------------*/

   if ((stencil_dim - 1) <= (relax_data -> memory_use))
   {
      hypre_SMGRelaxDestroyASol(relax_vdata);
   }

   hypre_EndTiming(relax_data -> time_index);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetup( void               *relax_vdata,
                     hypre_StructMatrix *A,
                     hypre_StructVector *b,
                     hypre_StructVector *x           )
{
   hypre_SMGRelaxData  *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   HYPRE_Int            stencil_dim;
   HYPRE_Int            a_sol_test;

   stencil_dim = hypre_StructStencilNDim(hypre_StructMatrixStencil(A));
   (relax_data -> stencil_dim) = stencil_dim;
   hypre_StructMatrixDestroy(relax_data -> A);
   hypre_StructVectorDestroy(relax_data -> b);
   hypre_StructVectorDestroy(relax_data -> x);
   (relax_data -> A) = hypre_StructMatrixRef(A);
   (relax_data -> b) = hypre_StructVectorRef(b);
   (relax_data -> x) = hypre_StructVectorRef(x);

   /*----------------------------------------------------------
    * Set up memory according to memory_use parameter.
    *
    * If a subset of the solver memory is not to be set up
    * until the solve is actually done, it's "setup" tag
    * should have a value greater than 1.
    *----------------------------------------------------------*/

   if ((stencil_dim - 1) <= (relax_data -> memory_use))
   {
      a_sol_test = 1;
   }
   else
   {
      a_sol_test = 0;
   }

   /*----------------------------------------------------------
    * Set up the solver
    *----------------------------------------------------------*/

   if ((relax_data -> setup_temp_vec) > 0)
   {
      hypre_SMGRelaxSetupTempVec(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_rem) > 0)
   {
      hypre_SMGRelaxSetupARem(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_sol) > a_sol_test)
   {
      hypre_SMGRelaxSetupASol(relax_vdata, A, b, x);
   }

   if ((relax_data -> base_box_array) == NULL)
   {
      hypre_SMGRelaxSetupBaseBoxArray(relax_vdata, A, b, x);
   }


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetupTempVec( void               *relax_vdata,
                            hypre_StructMatrix *A,
                            hypre_StructVector *b,
                            hypre_StructVector *x           )
{
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(x);

   hypre_SMGRelaxData  *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   hypre_StructVector  *temp_vec   = (relax_data -> temp_vec);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   if ((relax_data -> temp_vec) == NULL)
   {
      temp_vec = hypre_StructVectorCreate(hypre_StructVectorComm(b),
                                          hypre_StructVectorGrid(b));
      hypre_StructVectorSetNumGhost(temp_vec, hypre_StructVectorNumGhost(b));
      hypre_StructVectorInitialize(temp_vec);
      hypre_StructVectorAssemble(temp_vec);
      (relax_data -> temp_vec) = temp_vec;
   }
   (relax_data -> setup_temp_vec) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetupARem
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetupARem( void               *relax_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *b,
                         hypre_StructVector *x           )
{
   hypre_SMGRelaxData   *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   HYPRE_Int             num_spaces    = (relax_data -> num_spaces);
   HYPRE_Int            *space_indices = (relax_data -> space_indices);
   HYPRE_Int            *space_strides = (relax_data -> space_strides);
   hypre_StructVector   *temp_vec      = (relax_data -> temp_vec);

   hypre_StructStencil  *stencil       = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int             stencil_size  = hypre_StructStencilSize(stencil);
   HYPRE_Int             stencil_dim   = hypre_StructStencilNDim(stencil);

   hypre_StructMatrix   *A_rem;
   void                **residual_data;

   hypre_Index           base_index;
   hypre_Index           base_stride;

   HYPRE_Int             num_stencil_indices;
   HYPRE_Int            *stencil_indices;

   HYPRE_Int             i;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   hypre_SMGRelaxDestroyARem(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   hypre_CopyIndex((relax_data -> base_index),  base_index);
   hypre_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (hypre_IndexD(stencil_shape[i], (stencil_dim - 1)) != 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A_rem = hypre_StructMatrixCreateMask(A, num_stencil_indices, stencil_indices);
   hypre_TFree(stencil_indices, HYPRE_MEMORY_HOST);

   /* Set up residual_data */
   residual_data = hypre_TAlloc(void *,  num_spaces, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_spaces; i++)
   {
      hypre_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      hypre_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      residual_data[i] = hypre_SMGResidualCreate();
      hypre_SMGResidualSetBase(residual_data[i], base_index, base_stride);
      hypre_SMGResidualSetup(residual_data[i], A_rem, x, b, temp_vec);
   }

   (relax_data -> A_rem)         = A_rem;
   (relax_data -> residual_data) = residual_data;

   (relax_data -> setup_a_rem) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetupASol( void               *relax_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *b,
                         hypre_StructVector *x           )
{
   HYPRE_UNUSED_VAR(b);

   hypre_SMGRelaxData   *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   HYPRE_Int             num_spaces    = (relax_data -> num_spaces);
   HYPRE_Int            *space_indices = (relax_data -> space_indices);
   HYPRE_Int            *space_strides = (relax_data -> space_strides);
   hypre_StructVector   *temp_vec      = (relax_data -> temp_vec);

   HYPRE_Int             num_pre_relax   = (relax_data -> num_pre_relax);
   HYPRE_Int             num_post_relax  = (relax_data -> num_post_relax);

   hypre_StructStencil  *stencil       = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);
   HYPRE_Int             stencil_size  = hypre_StructStencilSize(stencil);
   HYPRE_Int             stencil_dim   = hypre_StructStencilNDim(stencil);

   hypre_StructMatrix   *A_sol;
   void                **solve_data;

   hypre_Index           base_index;
   hypre_Index           base_stride;

   HYPRE_Int             num_stencil_indices;
   HYPRE_Int            *stencil_indices;

   HYPRE_Int             i;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   hypre_SMGRelaxDestroyASol(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   hypre_CopyIndex((relax_data -> base_index),  base_index);
   hypre_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (hypre_IndexD(stencil_shape[i], (stencil_dim - 1)) == 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }

   A_sol = hypre_StructMatrixCreateMask(A, num_stencil_indices, stencil_indices);
   hypre_StructStencilNDim(hypre_StructMatrixStencil(A_sol)) = stencil_dim - 1;
   hypre_TFree(stencil_indices, HYPRE_MEMORY_HOST);

   /* Set up solve_data */
   solve_data    = hypre_TAlloc(void *,  num_spaces, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_spaces; i++)
   {
      hypre_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      hypre_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      if (stencil_dim > 2)
      {
         solve_data[i] = hypre_SMGCreate(relax_data -> comm);
         hypre_SMGSetNumPreRelax( solve_data[i], num_pre_relax);
         hypre_SMGSetNumPostRelax( solve_data[i], num_post_relax);
         hypre_SMGSetBase(solve_data[i], base_index, base_stride);
         hypre_SMGSetMemoryUse(solve_data[i], (relax_data -> memory_use));
         hypre_SMGSetTol(solve_data[i], 0.0);
         hypre_SMGSetMaxIter(solve_data[i], 1);
         hypre_StructSMGSetMaxLevel(solve_data[i], (relax_data -> max_level));
         hypre_SMGSetup(solve_data[i], A_sol, temp_vec, x);
      }
      else
      {
         solve_data[i] = hypre_CyclicReductionCreate(relax_data -> comm);
         hypre_CyclicReductionSetBase(solve_data[i], base_index, base_stride);
         //hypre_CyclicReductionSetMaxLevel(solve_data[i], -1);//(relax_data -> max_level)+10);
         hypre_CyclicReductionSetup(solve_data[i], A_sol, temp_vec, x);
      }
   }

   (relax_data -> A_sol)      = A_sol;
   (relax_data -> solve_data) = solve_data;

   (relax_data -> setup_a_sol) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetTempVec( void               *relax_vdata,
                          hypre_StructVector *temp_vec    )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   hypre_SMGRelaxDestroyTempVec(relax_vdata);
   (relax_data -> temp_vec) = hypre_StructVectorRef(temp_vec);

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetMemoryUse( void *relax_vdata,
                            HYPRE_Int   memory_use  )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> memory_use) = memory_use;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetTol( void   *relax_vdata,
                      HYPRE_Real  tol         )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> tol) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetMaxIter( void *relax_vdata,
                          HYPRE_Int   max_iter    )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetZeroGuess( void *relax_vdata,
                            HYPRE_Int   zero_guess  )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> zero_guess) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetNumSpaces( void *relax_vdata,
                            HYPRE_Int   num_spaces      )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   HYPRE_Int           i;

   (relax_data -> num_spaces) = num_spaces;

   hypre_TFree(relax_data -> space_indices, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> space_strides, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> pre_space_ranks, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> reg_space_ranks, HYPRE_MEMORY_HOST);
   (relax_data -> space_indices)   = hypre_TAlloc(HYPRE_Int,  num_spaces, HYPRE_MEMORY_HOST);
   (relax_data -> space_strides)   = hypre_TAlloc(HYPRE_Int,  num_spaces, HYPRE_MEMORY_HOST);
   (relax_data -> num_pre_spaces)  = 0;
   (relax_data -> num_reg_spaces)  = num_spaces;
   (relax_data -> pre_space_ranks) = NULL;
   (relax_data -> reg_space_ranks) = hypre_TAlloc(HYPRE_Int,  num_spaces, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_spaces; i++)
   {
      (relax_data -> space_indices[i]) = 0;
      (relax_data -> space_strides[i]) = 1;
      (relax_data -> reg_space_ranks[i]) = i;
   }

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetNumPreSpaces( void *relax_vdata,
                               HYPRE_Int   num_pre_spaces )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   HYPRE_Int           i;

   (relax_data -> num_pre_spaces) = num_pre_spaces;

   hypre_TFree(relax_data -> pre_space_ranks, HYPRE_MEMORY_HOST);
   (relax_data -> pre_space_ranks) = hypre_TAlloc(HYPRE_Int,  num_pre_spaces, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_pre_spaces; i++)
   {
      (relax_data -> pre_space_ranks[i]) = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetNumRegSpaces( void *relax_vdata,
                               HYPRE_Int   num_reg_spaces )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   HYPRE_Int           i;

   (relax_data -> num_reg_spaces) = num_reg_spaces;

   hypre_TFree(relax_data -> reg_space_ranks, HYPRE_MEMORY_HOST);
   (relax_data -> reg_space_ranks) = hypre_TAlloc(HYPRE_Int,  num_reg_spaces, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_reg_spaces; i++)
   {
      (relax_data -> reg_space_ranks[i]) = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetSpace( void *relax_vdata,
                        HYPRE_Int   i,
                        HYPRE_Int   space_index,
                        HYPRE_Int   space_stride )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> space_indices[i]) = space_index;
   (relax_data -> space_strides[i]) = space_stride;

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetRegSpaceRank( void *relax_vdata,
                               HYPRE_Int   i,
                               HYPRE_Int   reg_space_rank )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> reg_space_ranks[i]) = reg_space_rank;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetPreSpaceRank( void *relax_vdata,
                               HYPRE_Int   i,
                               HYPRE_Int   pre_space_rank  )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> pre_space_ranks[i]) = pre_space_rank;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetBase( void        *relax_vdata,
                       hypre_Index  base_index,
                       hypre_Index  base_stride )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;
   HYPRE_Int           d;

   for (d = 0; d < 3; d++)
   {
      hypre_IndexD((relax_data -> base_index),  d) =
         hypre_IndexD(base_index,  d);
      hypre_IndexD((relax_data -> base_stride), d) =
         hypre_IndexD(base_stride, d);
   }

   if ((relax_data -> base_box_array) != NULL)
   {
      hypre_BoxArrayDestroy((relax_data -> base_box_array));
      (relax_data -> base_box_array) = NULL;
   }

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetNumPreRelax( void *relax_vdata,
                              HYPRE_Int   num_pre_relax )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> num_pre_relax) = hypre_max(num_pre_relax, 1);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetNumPostRelax( void *relax_vdata,
                               HYPRE_Int   num_post_relax )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> num_post_relax) = num_post_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetNewMatrixStencil( void                *relax_vdata,
                                   hypre_StructStencil *diff_stencil )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   hypre_Index        *stencil_shape = hypre_StructStencilShape(diff_stencil);
   HYPRE_Int           stencil_size  = hypre_StructStencilSize(diff_stencil);
   HYPRE_Int           stencil_dim   = hypre_StructStencilNDim(diff_stencil);

   HYPRE_Int           i;

   for (i = 0; i < stencil_size; i++)
   {
      if (hypre_IndexD(stencil_shape[i], (stencil_dim - 1)) != 0)
      {
         (relax_data -> setup_a_rem) = 1;
      }
      else
      {
         (relax_data -> setup_a_sol) = 1;
      }
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetupBaseBoxArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetupBaseBoxArray( void               *relax_vdata,
                                 hypre_StructMatrix *A,
                                 hypre_StructVector *b,
                                 hypre_StructVector *x           )
{
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(b);

   hypre_SMGRelaxData  *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   hypre_StructGrid    *grid;
   hypre_BoxArray      *boxes;
   hypre_BoxArray      *base_box_array;

   grid  = hypre_StructVectorGrid(x);
   boxes = hypre_StructGridBoxes(grid);

   base_box_array = hypre_BoxArrayDuplicate(boxes);
   hypre_ProjectBoxArray(base_box_array,
                         (relax_data -> base_index),
                         (relax_data -> base_stride));

   (relax_data -> base_box_array) = base_box_array;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGRelaxSetMaxLevel( void *relax_vdata,
                           HYPRE_Int   num_max_level )
{
   hypre_SMGRelaxData *relax_data = (hypre_SMGRelaxData  *)relax_vdata;

   (relax_data -> max_level) = num_max_level;

   return hypre_error_flag;
}
