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
 * hypre_SMGRelaxData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   int                     setup_temp_vec;
   int                     setup_a_rem;
   int                     setup_a_sol;
                       
   MPI_Comm                comm;
                       
   int                     memory_use;
   double                  tol;
   int                     max_iter;
   int                     zero_guess;
                         
   int                     num_spaces;
   int                    *space_indices;
   int                    *space_strides;
                       
   int                     num_pre_spaces;
   int                     num_reg_spaces;
   int                    *pre_space_ranks;
   int                    *reg_space_ranks;

   hypre_Index             base_index;
   hypre_Index             base_stride;
   hypre_BoxArray         *base_box_array;

   int                     stencil_dim;
                       
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
   int                     num_iterations;
   int                     time_index;
                         
   int                     num_pre_relax;
   int                     num_post_relax;

} hypre_SMGRelaxData;

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SMGRelaxCreate( MPI_Comm  comm )
{
   hypre_SMGRelaxData *relax_data;

   relax_data = hypre_CTAlloc(hypre_SMGRelaxData, 1);
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
   (relax_data -> space_indices)      = hypre_TAlloc(int, 1);
   (relax_data -> space_strides)      = hypre_TAlloc(int, 1);
   (relax_data -> space_indices[0])   = 0;
   (relax_data -> space_strides[0])   = 1;
   (relax_data -> num_pre_spaces)     = 0;
   (relax_data -> num_reg_spaces)     = 1;
   (relax_data -> pre_space_ranks)    = NULL;
   (relax_data -> reg_space_ranks)    = hypre_TAlloc(int, 1);
   (relax_data -> reg_space_ranks[0]) = 0;
   hypre_SetIndex((relax_data -> base_index), 0, 0, 0);
   hypre_SetIndex((relax_data -> base_stride), 1, 1, 1);
   (relax_data -> A)                  = NULL;
   (relax_data -> b)                  = NULL;
   (relax_data -> x)                  = NULL;
   (relax_data -> temp_vec)           = NULL;

   (relax_data -> num_pre_relax) = 1;
   (relax_data -> num_post_relax) = 1;

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxDestroyTempVec
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxDestroyTempVec( void *relax_vdata )
{
   hypre_SMGRelaxData  *relax_data = relax_vdata;
   int                  ierr = 0;

   hypre_StructVectorDestroy(relax_data -> temp_vec);
   (relax_data -> setup_temp_vec) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxDestroyARem
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxDestroyARem( void *relax_vdata )
{
   hypre_SMGRelaxData  *relax_data = relax_vdata;
   int                  i;
   int                  ierr = 0;

   if (relax_data -> A_rem)
   {
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         hypre_SMGResidualDestroy(relax_data -> residual_data[i]);
      }
      hypre_TFree(relax_data -> residual_data);
      hypre_StructMatrixDestroy(relax_data -> A_rem);
      (relax_data -> A_rem) = NULL;
   }
   (relax_data -> setup_a_rem) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxDestroyASol
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxDestroyASol( void *relax_vdata )
{
   hypre_SMGRelaxData  *relax_data = relax_vdata;
   int                  stencil_dim;
   int                  i;
   int                  ierr = 0;

   if (relax_data -> A_sol)
   {
      stencil_dim = (relax_data -> stencil_dim);
      for (i = 0; i < (relax_data -> num_spaces); i++)
      {
         if (stencil_dim > 2)
            hypre_SMGDestroy(relax_data -> solve_data[i]);
         else
            hypre_CyclicReductionDestroy(relax_data -> solve_data[i]);
      }
      hypre_TFree(relax_data -> solve_data);
      hypre_StructMatrixDestroy(relax_data -> A_sol);
      (relax_data -> A_sol) = NULL;
   }
   (relax_data -> setup_a_sol) = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxDestroy
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxDestroy( void *relax_vdata )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   if (relax_data)
   {
      hypre_TFree(relax_data -> space_indices);
      hypre_TFree(relax_data -> space_strides);
      hypre_TFree(relax_data -> pre_space_ranks);
      hypre_TFree(relax_data -> reg_space_ranks);
      hypre_BoxArrayDestroy(relax_data -> base_box_array);

      hypre_StructMatrixDestroy(relax_data -> A);
      hypre_StructVectorDestroy(relax_data -> b);
      hypre_StructVectorDestroy(relax_data -> x);

      hypre_SMGRelaxDestroyTempVec(relax_vdata);
      hypre_SMGRelaxDestroyARem(relax_vdata);
      hypre_SMGRelaxDestroyASol(relax_vdata);

      hypre_FinalizeTiming(relax_data -> time_index);
      hypre_TFree(relax_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelax
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelax( void               *relax_vdata,
                hypre_StructMatrix *A,
                hypre_StructVector *b,
                hypre_StructVector *x           )
{
   hypre_SMGRelaxData   *relax_data = relax_vdata;

   int                   zero_guess;
   int                   stencil_dim;
   hypre_StructVector   *temp_vec;
   hypre_StructMatrix   *A_sol;
   hypre_StructMatrix   *A_rem;
   void                **residual_data;
   void                **solve_data;

   hypre_IndexRef        base_stride;
   hypre_BoxArray       *base_box_a;
   double                zero = 0.0;

   int                   max_iter;
   int                   num_spaces;
   int                  *space_ranks;
                    
   int                   i, j, k, is;
                    
   int                   ierr = 0;

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
      ierr = hypre_SMGSetStructVectorConstantValues(x, zero, base_box_a,
                                                    base_stride); 
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

            hypre_SMGResidual(residual_data[is], A_rem, x, b, temp_vec);

            if (stencil_dim > 2)
               hypre_SMGSolve(solve_data[is], A_sol, temp_vec, x);
            else
               hypre_CyclicReduction(solve_data[is], A_sol, temp_vec, x);
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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetup
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetup( void               *relax_vdata,
                     hypre_StructMatrix *A,
                     hypre_StructVector *b,
                     hypre_StructVector *x           )
{
   hypre_SMGRelaxData  *relax_data = relax_vdata;
   int                  stencil_dim;
   int                  a_sol_test;
   int                  ierr = 0;

   stencil_dim = hypre_StructStencilDim(hypre_StructMatrixStencil(A));
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
      ierr = hypre_SMGRelaxSetupTempVec(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_rem) > 0)
   {
      ierr = hypre_SMGRelaxSetupARem(relax_vdata, A, b, x);
   }

   if ((relax_data -> setup_a_sol) > a_sol_test)
   {
      ierr = hypre_SMGRelaxSetupASol(relax_vdata, A, b, x);
   }

   if ((relax_data -> base_box_array) == NULL)
   {
      ierr = hypre_SMGRelaxSetupBaseBoxArray(relax_vdata, A, b, x);
   }
   

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetupTempVec
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetupTempVec( void               *relax_vdata,
                            hypre_StructMatrix *A,
                            hypre_StructVector *b,
                            hypre_StructVector *x           )
{
   hypre_SMGRelaxData  *relax_data = relax_vdata;
   hypre_StructVector  *temp_vec   = (relax_data -> temp_vec);
   int                  ierr = 0;

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetupARem
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetupARem( void               *relax_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *b,
                         hypre_StructVector *x           )
{
   hypre_SMGRelaxData   *relax_data = relax_vdata;

   int                   num_spaces    = (relax_data -> num_spaces);
   int                  *space_indices = (relax_data -> space_indices);
   int                  *space_strides = (relax_data -> space_strides);
   hypre_StructVector   *temp_vec      = (relax_data -> temp_vec);

   hypre_StructStencil  *stencil       = hypre_StructMatrixStencil(A);     
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);
   int                   stencil_size  = hypre_StructStencilSize(stencil); 
   int                   stencil_dim   = hypre_StructStencilDim(stencil);
                       
   hypre_StructMatrix   *A_rem;
   void                **residual_data;

   hypre_Index           base_index;
   hypre_Index           base_stride;

   int                   num_stencil_indices;
   int                  *stencil_indices;
                       
   int                   i;

   int                   ierr = 0;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   hypre_SMGRelaxDestroyARem(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   hypre_CopyIndex((relax_data -> base_index),  base_index);
   hypre_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = hypre_TAlloc(int, stencil_size);
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
   hypre_TFree(stencil_indices);

   /* Set up residual_data */
   residual_data = hypre_TAlloc(void *, num_spaces);

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetupASol
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetupASol( void               *relax_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *b,
                         hypre_StructVector *x           )
{
   hypre_SMGRelaxData   *relax_data = relax_vdata;

   int                   num_spaces    = (relax_data -> num_spaces);
   int                  *space_indices = (relax_data -> space_indices);
   int                  *space_strides = (relax_data -> space_strides);
   hypre_StructVector   *temp_vec      = (relax_data -> temp_vec);

   int                   num_pre_relax   = (relax_data -> num_pre_relax);
   int                   num_post_relax  = (relax_data -> num_post_relax);

   hypre_StructStencil  *stencil       = hypre_StructMatrixStencil(A);     
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);
   int                   stencil_size  = hypre_StructStencilSize(stencil); 
   int                   stencil_dim   = hypre_StructStencilDim(stencil);
                       
   hypre_StructMatrix   *A_sol;
   void                **solve_data;

   hypre_Index           base_index;
   hypre_Index           base_stride;

   int                   num_stencil_indices;
   int                  *stencil_indices;
                       
   int                   i;

   int                   ierr = 0;

   /*----------------------------------------------------------
    * Free up old data before putting new data into structure
    *----------------------------------------------------------*/

   hypre_SMGRelaxDestroyASol(relax_vdata);

   /*----------------------------------------------------------
    * Set up data
    *----------------------------------------------------------*/

   hypre_CopyIndex((relax_data -> base_index),  base_index);
   hypre_CopyIndex((relax_data -> base_stride), base_stride);

   stencil_indices = hypre_TAlloc(int, stencil_size);
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
   hypre_StructStencilDim(hypre_StructMatrixStencil(A_sol)) = stencil_dim - 1;
   hypre_TFree(stencil_indices);

   /* Set up solve_data */
   solve_data    = hypre_TAlloc(void *, num_spaces);

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
         hypre_SMGSetup(solve_data[i], A_sol, temp_vec, x);
      }
      else
      {
         solve_data[i] = hypre_CyclicReductionCreate(relax_data -> comm);
         hypre_CyclicReductionSetBase(solve_data[i], base_index, base_stride);
         hypre_CyclicReductionSetup(solve_data[i], A_sol, temp_vec, x);
      }
   }

   (relax_data -> A_sol)      = A_sol;
   (relax_data -> solve_data) = solve_data;

   (relax_data -> setup_a_sol) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetTempVec
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetTempVec( void               *relax_vdata,
                          hypre_StructVector *temp_vec    )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   hypre_SMGRelaxDestroyTempVec(relax_vdata);
   (relax_data -> temp_vec) = hypre_StructVectorRef(temp_vec);

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetMemoryUse
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetMemoryUse( void *relax_vdata,
                            int   memory_use  )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int               ierr = 0;

   (relax_data -> memory_use) = memory_use;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetTol
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetTol( void   *relax_vdata,
                      double  tol         )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetMaxIter
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetMaxIter( void *relax_vdata,
                          int   max_iter    )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> max_iter) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetZeroGuess
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetZeroGuess( void *relax_vdata,
                            int   zero_guess  )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> zero_guess) = zero_guess;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetNumSpaces
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetNumSpaces( void *relax_vdata,
                            int   num_spaces      )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 i;
   int                 ierr = 0;

   (relax_data -> num_spaces) = num_spaces;

   hypre_TFree(relax_data -> space_indices);
   hypre_TFree(relax_data -> space_strides);
   hypre_TFree(relax_data -> pre_space_ranks);
   hypre_TFree(relax_data -> reg_space_ranks);
   (relax_data -> space_indices)   = hypre_TAlloc(int, num_spaces);
   (relax_data -> space_strides)   = hypre_TAlloc(int, num_spaces);
   (relax_data -> num_pre_spaces)  = 0;
   (relax_data -> num_reg_spaces)  = num_spaces;
   (relax_data -> pre_space_ranks) = NULL;
   (relax_data -> reg_space_ranks) = hypre_TAlloc(int, num_spaces);

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
 * hypre_SMGRelaxSetNumPreSpaces
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetNumPreSpaces( void *relax_vdata,
                               int   num_pre_spaces )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 i;
   int                 ierr = 0;

   (relax_data -> num_pre_spaces) = num_pre_spaces;

   hypre_TFree(relax_data -> pre_space_ranks);
   (relax_data -> pre_space_ranks) = hypre_TAlloc(int, num_pre_spaces);

   for (i = 0; i < num_pre_spaces; i++)
      (relax_data -> pre_space_ranks[i]) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetNumRegSpaces
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetNumRegSpaces( void *relax_vdata,
                               int   num_reg_spaces )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 i;
   int                 ierr = 0;

   (relax_data -> num_reg_spaces) = num_reg_spaces;

   hypre_TFree(relax_data -> reg_space_ranks);
   (relax_data -> reg_space_ranks) = hypre_TAlloc(int, num_reg_spaces);

   for (i = 0; i < num_reg_spaces; i++)
      (relax_data -> reg_space_ranks[i]) = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetSpace
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetSpace( void *relax_vdata,
                        int   i,
                        int   space_index,
                        int   space_stride )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> space_indices[i]) = space_index;
   (relax_data -> space_strides[i]) = space_stride;

   (relax_data -> setup_temp_vec) = 1;
   (relax_data -> setup_a_rem)    = 1;
   (relax_data -> setup_a_sol)    = 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetRegSpaceRank
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetRegSpaceRank( void *relax_vdata,
                               int   i,
                               int   reg_space_rank )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> reg_space_ranks[i]) = reg_space_rank;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetPreSpaceRank
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetPreSpaceRank( void *relax_vdata,
                               int   i,
                               int   pre_space_rank  )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> pre_space_ranks[i]) = pre_space_rank;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetBase
 *--------------------------------------------------------------------------*/
 
int
hypre_SMGRelaxSetBase( void        *relax_vdata,
                       hypre_Index  base_index,
                       hypre_Index  base_stride )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 d;
   int                 ierr = 0;
 
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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetNumPreRelax
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetNumPreRelax( void *relax_vdata,
                              int   num_pre_relax )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> num_pre_relax) = hypre_max(num_pre_relax,1);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetNumPostRelax
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetNumPostRelax( void *relax_vdata,
                               int   num_post_relax )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;
   int                 ierr = 0;

   (relax_data -> num_post_relax) = num_post_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetNewMatrixStencil
 *--------------------------------------------------------------------------*/
 
int
hypre_SMGRelaxSetNewMatrixStencil( void                *relax_vdata,
                                   hypre_StructStencil *diff_stencil )
{
   hypre_SMGRelaxData *relax_data = relax_vdata;

   hypre_Index        *stencil_shape = hypre_StructStencilShape(diff_stencil);
   int                 stencil_size  = hypre_StructStencilSize(diff_stencil); 
   int                 stencil_dim   = hypre_StructStencilDim(diff_stencil);
                         
   int                 i;
                     
   int                 ierr = 0;

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

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_SMGRelaxSetupBaseBoxArray
 *--------------------------------------------------------------------------*/

int
hypre_SMGRelaxSetupBaseBoxArray( void               *relax_vdata,
                                 hypre_StructMatrix *A,
                                 hypre_StructVector *b,
                                 hypre_StructVector *x           )
{
   hypre_SMGRelaxData  *relax_data = relax_vdata;
                       
   hypre_StructGrid    *grid;
   hypre_BoxArray      *boxes;
   hypre_BoxArray      *base_box_array;
                       
   int                  ierr = 0;

   grid  = hypre_StructVectorGrid(x);
   boxes = hypre_StructGridBoxes(grid);

   base_box_array = hypre_BoxArrayDuplicate(boxes);
   hypre_ProjectBoxArray(base_box_array, 
                         (relax_data -> base_index),
                         (relax_data -> base_stride));

   (relax_data -> base_box_array) = base_box_array;

   return ierr;
}

