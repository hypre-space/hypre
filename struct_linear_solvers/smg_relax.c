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
   int                   setup_temp_vec;
   int                   setup_a_rem;
   int                   setup_a_sol;

   MPI_Comm             *comm;

   int                   memory_use;
   double                tol;
   int                   max_iter;
   int                   zero_guess;
                       
   int                   num_spaces;
   int                  *space_indices;
   int                  *space_strides;

   int                   num_pre_spaces;
   int                   num_reg_spaces;
   int                  *pre_space_ranks;
   int                  *reg_space_ranks;

   zzz_Index            *base_index;
   zzz_Index            *base_stride;
   zzz_SBoxArray        *base_sbox_array;

   int                   stencil_dim;
                       
   zzz_StructMatrix     *A;
   zzz_StructVector     *b;
   zzz_StructVector     *x;

   zzz_StructVector     *temp_vec;
   int                   temp_vec_allocated;
   zzz_StructMatrix     *A_sol;       /* Coefficients of A that make up
                                         the (sol)ve part of the relaxation */
   zzz_StructMatrix     *A_rem;       /* Coefficients of A (rem)aining:
                                         A_rem = A - A_sol                  */
   void                **residual_data;  /* Array of size `num_spaces' */
   void                **solve_data;     /* Array of size `num_spaces' */

   /* log info (always logged) */
   int                   num_iterations;
   int                   time_index;

   int                   num_pre_relax;
   int                   num_post_relax;

} zzz_SMGRelaxData;

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGRelaxInitialize( MPI_Comm *comm )
{
   zzz_SMGRelaxData *relax_data;

   relax_data = zzz_CTAlloc(zzz_SMGRelaxData, 1);

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;
   (relax_data -> comm)           = comm;
   (relax_data -> base_index)     = zzz_NewIndex();
   (relax_data -> base_stride)    = zzz_NewIndex();
   (relax_data -> base_sbox_array)= NULL;
   (relax_data -> time_index)     = zzz_InitializeTiming("SMGRelax");

   /* set defaults */
   (relax_data -> memory_use)         = 0;
   (relax_data -> tol)                = 1.0e-06;
   (relax_data -> max_iter)           = 1000;
   (relax_data -> zero_guess)         = 0;
   (relax_data -> num_spaces)         = 1;
   (relax_data -> space_indices)      = zzz_TAlloc(int, 1);
   (relax_data -> space_strides)      = zzz_TAlloc(int, 1);
   (relax_data -> space_indices[0])   = 0;
   (relax_data -> space_strides[0])   = 1;
   (relax_data -> num_pre_spaces)     = 0;
   (relax_data -> num_reg_spaces)     = 1;
   (relax_data -> pre_space_ranks)    = NULL;
   (relax_data -> reg_space_ranks)    = zzz_TAlloc(int, 1);
   (relax_data -> reg_space_ranks[0]) = 0;
   zzz_SetIndex((relax_data -> base_index), 0, 0, 0);
   zzz_SetIndex((relax_data -> base_stride), 1, 1, 1);
   (relax_data -> temp_vec)           = NULL;
   (relax_data -> temp_vec_allocated) = 1;

   (relax_data -> num_pre_relax) = 1;
   (relax_data -> num_post_relax) = 1;

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxFreeTempVec
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxFreeTempVec( void *relax_vdata )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;
   int                   ierr;

   if (relax_data -> temp_vec_allocated)
   {
      zzz_FreeStructVector(relax_data -> temp_vec);
   }
   (relax_data -> setup_temp_vec) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxFreeARem
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxFreeARem( void *relax_vdata )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;
   int                   i;
   int                   ierr;

   if (relax_data -> A_rem)
   {
      zzz_FreeStructMatrixMask(relax_data -> A_rem);
      (relax_data -> A_rem) = NULL;
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         zzz_SMGResidualFinalize(relax_data -> residual_data[i]);
      }
      zzz_TFree(relax_data -> residual_data);
   }
   (relax_data -> setup_a_rem) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxFreeASol
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxFreeASol( void *relax_vdata )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;
   int                   stencil_dim;
   int                   i;
   int                   ierr;

   if (relax_data -> A_sol)
   {
      stencil_dim = (relax_data -> stencil_dim);
      zzz_FreeStructMatrixMask(relax_data -> A_sol);
      (relax_data -> A_sol) = NULL;
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         if (stencil_dim > 2)
            zzz_SMGFinalize(relax_data -> solve_data[i]);
         else
            zzz_CyclicReductionFinalize(relax_data -> solve_data[i]);
      }
      zzz_TFree(relax_data -> solve_data);
   }
   (relax_data -> setup_a_sol) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxFinalize( void *relax_vdata )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr;

   if (relax_data)
   {
      zzz_TFree(relax_data -> space_indices);
      zzz_TFree(relax_data -> space_strides);
      zzz_TFree(relax_data -> pre_space_ranks);
      zzz_TFree(relax_data -> reg_space_ranks);
      zzz_FreeIndex(relax_data -> base_index);
      zzz_FreeIndex(relax_data -> base_stride);
      zzz_FreeSBoxArray(relax_data -> base_sbox_array);

      zzz_SMGRelaxFreeTempVec(relax_vdata);
      zzz_SMGRelaxFreeARem(relax_vdata);
      zzz_SMGRelaxFreeASol(relax_vdata);

      zzz_FinalizeTiming(relax_data -> time_index);
      zzz_TFree(relax_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelax
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelax( void             *relax_vdata,
              zzz_StructMatrix *A,
              zzz_StructVector *b,
              zzz_StructVector *x               )
{
   zzz_SMGRelaxData   *relax_data = relax_vdata;

   int                 zero_guess;
   int                 stencil_dim;
   zzz_StructVector   *temp_vec;
   zzz_StructMatrix   *A_sol;
   zzz_StructMatrix   *A_rem;
   void              **residual_data;
   void              **solve_data;

   zzz_SBoxArray      *base_sbox_a;
   double              zero = 0.0;

   int                 max_iter;
   int                 num_spaces;
   int                *space_ranks;

   int                 i, j, k, is;

   int                 ierr;

   /*----------------------------------------------------------
    * Note: The zero_guess stuff is not handled correctly
    * for general relaxation parameters.  It is correct when
    * the spaces are independent sets in the direction of
    * relaxation.
    *----------------------------------------------------------*/

   zzz_BeginTiming(relax_data -> time_index);

   /*----------------------------------------------------------
    * Set up the solver
    *----------------------------------------------------------*/

   /* insure that the solver memory gets fully set up */
   if ((relax_data -> setup_a_sol) > 0)
   {
      (relax_data -> setup_a_sol) = 2;
   }

   zzz_SMGRelaxSetup(relax_vdata, A, b, x);

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
      base_sbox_a = (relax_data -> base_sbox_array);
      ierr = zzz_SMGSetStructVectorConstantValues(x, zero, base_sbox_a); 
   }

   /*----------------------------------------------------------
    * Iterate
    *----------------------------------------------------------*/

   for (k = 0; k < 2; k++)
   {
      switch(k)
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

            zzz_SMGResidual(residual_data[is], A_rem, x, b, temp_vec);

            if (stencil_dim > 2)
               zzz_SMGSolve(solve_data[is], A_sol, temp_vec, x);
            else
               zzz_CyclicReduction(solve_data[is], A_sol, temp_vec, x);
         }

         (relax_data -> num_iterations) = (i + 1);
      }
   }

   /*----------------------------------------------------------
    * Free up memory according to memory_use parameter
    *----------------------------------------------------------*/

   if ((stencil_dim - 1) <= (relax_data -> memory_use))
   {
      zzz_SMGRelaxFreeASol(relax_vdata);
   }

   zzz_EndTiming(relax_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetup( void             *relax_vdata,
                   zzz_StructMatrix *A,
                   zzz_StructVector *b,
                   zzz_StructVector *x           )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;
   int                   stencil_dim;
   int                   a_sol_test;
   int                   ierr;

   stencil_dim = zzz_StructStencilDim(zzz_StructMatrixStencil(A));
   (relax_data -> stencil_dim) = stencil_dim;
   (relax_data -> A)           = A;
   (relax_data -> b)           = b;
   (relax_data -> x)           = x;

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
      ierr = zzz_SMGRelaxSetupTempVec(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_rem) > 0)
   {
      ierr = zzz_SMGRelaxSetupARem(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_sol) > a_sol_test)
   {
      ierr = zzz_SMGRelaxSetupASol(relax_vdata, A, b, x);
   }

   if ((relax_data -> base_sbox_array) == NULL)
   {
      ierr = zzz_SMGRelaxSetupBaseSBoxArray(relax_vdata, A, b, x);
   }
   

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetupTempVec
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetupTempVec( void             *relax_vdata,
                          zzz_StructMatrix *A,
                          zzz_StructVector *b,
                          zzz_StructVector *x           )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;
   zzz_StructVector     *temp_vec      = (relax_data -> temp_vec);
   int                   ierr;

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   if (relax_data -> temp_vec_allocated)
   {
      temp_vec = zzz_NewStructVector(zzz_StructVectorComm(b),
                                     zzz_StructVectorGrid(b));
      zzz_SetStructVectorNumGhost(temp_vec, zzz_StructVectorNumGhost(b));
      zzz_InitializeStructVector(temp_vec);
      zzz_AssembleStructVector(temp_vec);
      (relax_data -> temp_vec)           = temp_vec;
      (relax_data -> temp_vec_allocated) = 1;
   }
   (relax_data -> setup_temp_vec) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetupARem
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetupARem( void             *relax_vdata,
                       zzz_StructMatrix *A,
                       zzz_StructVector *b,
                       zzz_StructVector *x           )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;

   int                   num_spaces    = (relax_data -> num_spaces);
   int                  *space_indices = (relax_data -> space_indices);
   int                  *space_strides = (relax_data -> space_strides);
   zzz_StructVector     *temp_vec      = (relax_data -> temp_vec);

   zzz_StructStencil    *stencil       = zzz_StructMatrixStencil(A);     
   zzz_Index           **stencil_shape = zzz_StructStencilShape(stencil);
   int                   stencil_size  = zzz_StructStencilSize(stencil); 
   int                   stencil_dim   = zzz_StructStencilDim(stencil);
                       
   zzz_StructMatrix     *A_rem;
   void                **residual_data;

   zzz_Index            *base_index;
   zzz_Index            *base_stride;

   int                   num_stencil_indices;
   int                  *stencil_indices;
                       
   int                   i;

   int                   ierr;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   zzz_SMGRelaxFreeARem(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   base_index  = zzz_NewIndex();
   base_stride = zzz_NewIndex();
   zzz_CopyIndex((relax_data -> base_index),  base_index);
   zzz_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = zzz_TAlloc(int, stencil_size);
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
   zzz_TFree(stencil_indices);

   /* Set up residual_data */
   residual_data = zzz_TAlloc(void *, num_spaces);
   for (i = 0; i < num_spaces; i++)
   {
      zzz_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      zzz_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      residual_data[i] = zzz_SMGResidualInitialize();
      zzz_SMGResidualSetBase(residual_data[i], base_index, base_stride);
      zzz_SMGResidualSetup(residual_data[i], A_rem, x, b, temp_vec);
   }

   zzz_FreeIndex(base_index);
   zzz_FreeIndex(base_stride);

   (relax_data -> A_rem)         = A_rem;
   (relax_data -> residual_data) = residual_data;

   (relax_data -> setup_a_rem) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetupASol
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetupASol( void             *relax_vdata,
                       zzz_StructMatrix *A,
                       zzz_StructVector *b,
                       zzz_StructVector *x           )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;

   int                   num_spaces    = (relax_data -> num_spaces);
   int                  *space_indices = (relax_data -> space_indices);
   int                  *space_strides = (relax_data -> space_strides);
   zzz_StructVector     *temp_vec      = (relax_data -> temp_vec);

   int                   num_pre_relax   = (relax_data -> num_pre_relax);
   int                   num_post_relax  = (relax_data -> num_post_relax);

   zzz_StructStencil    *stencil       = zzz_StructMatrixStencil(A);     
   zzz_Index           **stencil_shape = zzz_StructStencilShape(stencil);
   int                   stencil_size  = zzz_StructStencilSize(stencil); 
   int                   stencil_dim   = zzz_StructStencilDim(stencil);
                       
   zzz_StructMatrix     *A_sol;
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

   zzz_SMGRelaxFreeASol(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   base_index  = zzz_NewIndex();
   base_stride = zzz_NewIndex();
   zzz_CopyIndex((relax_data -> base_index),  base_index);
   zzz_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = zzz_TAlloc(int, stencil_size);
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
   zzz_TFree(stencil_indices);

   /* Set up solve_data */
   solve_data    = zzz_TAlloc(void *, num_spaces);
   for (i = 0; i < num_spaces; i++)
   {
      zzz_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      zzz_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      if (stencil_dim > 2)
      {
         solve_data[i] = zzz_SMGInitialize(relax_data -> comm);
         zzz_SMGSetNumPreRelax( solve_data[i], num_pre_relax);
         zzz_SMGSetNumPostRelax( solve_data[i], num_post_relax);
         zzz_SMGSetBase(solve_data[i], base_index, base_stride);
         zzz_SMGSetMemoryUse(solve_data[i], (relax_data -> memory_use));
         zzz_SMGSetTol(solve_data[i], 0.0);
         zzz_SMGSetMaxIter(solve_data[i], 1);
         zzz_SMGSetup(solve_data[i], A_sol, temp_vec, x);
      }
      else
      {
         solve_data[i] = zzz_CyclicReductionInitialize(relax_data -> comm);
         zzz_CyclicReductionSetBase(solve_data[i], base_index, base_stride);
         zzz_CyclicReductionSetup(solve_data[i], A_sol, temp_vec, x);
      }
   }

   zzz_FreeIndex(base_index);
   zzz_FreeIndex(base_stride);

   (relax_data -> A_sol)      = A_sol;
   (relax_data -> solve_data) = solve_data;

   (relax_data -> setup_a_sol) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetTempVec
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetTempVec( void             *relax_vdata,
                        zzz_StructVector *temp_vec    )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   zzz_SMGRelaxFreeTempVec(relax_vdata);
   (relax_data -> temp_vec)           = temp_vec;
   (relax_data -> temp_vec_allocated) = 0;

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetMemoryUse
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetMemoryUse( void *relax_vdata,
                          int   memory_use  )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> memory_use) = memory_use;

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
 * zzz_SMGRelaxSetNonZeroGuess
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetNonZeroGuess( void *relax_vdata )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> zero_guess) = 0;

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
   zzz_TFree(relax_data -> pre_space_ranks);
   zzz_TFree(relax_data -> reg_space_ranks);
   (relax_data -> space_indices)   = zzz_TAlloc(int, num_spaces);
   (relax_data -> space_strides)   = zzz_TAlloc(int, num_spaces);
   (relax_data -> num_pre_spaces)  = 0;
   (relax_data -> num_reg_spaces)  = num_spaces;
   (relax_data -> pre_space_ranks) = NULL;
   (relax_data -> reg_space_ranks) = zzz_TAlloc(int, num_spaces);

   for (i = 0; i < num_spaces; i++)
   {
      (relax_data -> space_indices[i]) = 0;
      (relax_data -> space_strides[i]) = 1;
      (relax_data -> reg_space_ranks[i]) = i;
   }

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetNumPreSpaces
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetNumPreSpaces( void *relax_vdata,
                             int   num_pre_spaces )
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
 * zzz_SMGRelaxSetNumRegSpaces
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetNumRegSpaces( void *relax_vdata,
                             int   num_reg_spaces )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               i;
   int               ierr = 0;

   (relax_data -> num_reg_spaces) = num_reg_spaces;

   zzz_TFree(relax_data -> reg_space_ranks);
   (relax_data -> reg_space_ranks) = zzz_TAlloc(int, num_reg_spaces);

   for (i = 0; i < num_reg_spaces; i++)
      (relax_data -> reg_space_ranks[i]) = 0;

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

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetRegSpaceRank
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetRegSpaceRank( void *relax_vdata,
                             int   i,
                             int   reg_space_rank )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> reg_space_ranks[i]) = reg_space_rank;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetPreSpaceRank
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetPreSpaceRank( void *relax_vdata,
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
   int               d;
   int               ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD((relax_data -> base_index),  d) =
         zzz_IndexD(base_index,  d);
      zzz_IndexD((relax_data -> base_stride), d) =
         zzz_IndexD(base_stride, d);
   }
 
   if ((relax_data -> base_sbox_array) != NULL)
   {
      zzz_FreeSBoxArray((relax_data -> base_sbox_array));
      (relax_data -> base_sbox_array) = NULL;
   }

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetNumPreRelax
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetNumPreRelax( void *relax_vdata,
                            int   num_pre_relax )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> num_pre_relax) = max(num_pre_relax,1);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetNumPostRelax( void *relax_vdata,
                             int   num_post_relax )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;
   int          ierr = 0;

   (relax_data -> num_post_relax) = num_post_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetNewMatrixStencil
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGRelaxSetNewMatrixStencil( void              *relax_vdata,
                                 zzz_StructStencil *diff_stencil )
{
   zzz_SMGRelaxData *relax_data = relax_vdata;

   zzz_Index       **stencil_shape = zzz_StructStencilShape(diff_stencil);
   int               stencil_size  = zzz_StructStencilSize(diff_stencil); 
   int               stencil_dim   = zzz_StructStencilDim(diff_stencil);
                       
   int               i;

   int               ierr = 0;

   for (i = 0; i < stencil_size; i++)
   {
      if (zzz_IndexD(stencil_shape[i], (stencil_dim - 1)) != 0)
      {
         (relax_data -> setup_a_rem) = 1;
      }
      else
      {
         (relax_data -> setup_a_sol) = 1;
      }
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetupBaseSBoxArray
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetupBaseSBoxArray( void             *relax_vdata,
                                zzz_StructMatrix *A,
                                zzz_StructVector *b,
                                zzz_StructVector *x           )
{
   zzz_SMGRelaxData     *relax_data = relax_vdata;

   zzz_StructGrid       *grid;
   zzz_BoxArray         *boxes;
   zzz_SBoxArray        *sboxes;

   int                   ierr;

   grid  = zzz_StructVectorGrid(x);
   boxes = zzz_StructGridBoxes(grid);

   sboxes = zzz_ProjectBoxArray( boxes, 
                                 (relax_data -> base_index),
                                 (relax_data -> base_stride) );

   (relax_data -> base_sbox_array) = sboxes;

   return ierr;
}

