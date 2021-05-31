/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

#define MAX_DEPTH 10

typedef struct hypre_SSAMGRelaxData_struct
{
   MPI_Comm                comm;

   HYPRE_Int               nparts;
   HYPRE_Int               type;
   HYPRE_Int               max_iter;
   HYPRE_Int               zero_guess;
   HYPRE_Real              tol;
   HYPRE_Real             *weights;  /* nparts array */
   HYPRE_Real             *mweights; /* nparts array */
   HYPRE_Int              *active_p; /* active parts */

   hypre_SStructMatrix    *A;
   hypre_SStructVector    *b;
   hypre_SStructVector    *x;
   hypre_SStructVector    *t;

   /* defines set of nodes to apply relaxation */
   HYPRE_Int               num_nodesets;
   HYPRE_Int              *nodeset_sizes;   /* (num_nodeset) */
   HYPRE_Int              *nodeset_ranks;   /* (num_nodeset) */
   hypre_Index            *nodeset_strides; /* (num_nodeset) */
   hypre_Index           **nodeset_indices; /* (num_nodeset x nodeset_size) */

   /* defines sends and recieves for each StructVector */
   hypre_ComputePkg    ****svec_compute_pkgs; /* (nparts x num_nodeset) */
   hypre_CommHandle     ***comm_handle;       /* (nparts x num_nodeset) */

   /* defines independent and dependent boxes for computations */
   hypre_ComputePkg     ***compute_pkgs; /* (nparts x num_nodeset) */

   /* pointers to local storage used to invert diagonal blocks */
   HYPRE_Real            **A_loc;
   HYPRE_Real             *x_loc;

   /* pointers for vector and matrix data */
   HYPRE_Real           ***Ap;
   HYPRE_Real            **bp;
   HYPRE_Real            **xp;
   HYPRE_Real            **tp;

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;

   /* Matvec data structure */
   void                   *matvec_vdata;

} hypre_SSAMGRelaxData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxCreate( MPI_Comm    comm,
                        HYPRE_Int   nparts,
                        void      **relax_vdata_ptr)
{
   hypre_SSAMGRelaxData  *relax_data;

   relax_data = hypre_CTAlloc(hypre_SSAMGRelaxData, 1, HYPRE_MEMORY_HOST);

   (relax_data -> comm)       = comm;
   (relax_data -> time_index) = hypre_InitializeTiming("SSAMG Relax");

   /* set defaults */
   (relax_data -> nparts)            = nparts;
   (relax_data -> tol)               = 1.0e-6;
   (relax_data -> max_iter)          = 1;
   (relax_data -> zero_guess)        = 0;
   (relax_data -> type)              = 0;
   (relax_data -> weights)           = NULL;
   (relax_data -> mweights)          = NULL;
   (relax_data -> active_p)          = NULL;
   (relax_data -> A)                 = NULL;
   (relax_data -> b)                 = NULL;
   (relax_data -> x)                 = NULL;
   (relax_data -> t)                 = NULL;
   (relax_data -> num_nodesets)      = 0;
   (relax_data -> nodeset_sizes)     = NULL;
   (relax_data -> nodeset_ranks)     = NULL;
   (relax_data -> nodeset_strides)   = NULL;
   (relax_data -> nodeset_indices)   = NULL;
   (relax_data -> svec_compute_pkgs) = NULL;
   (relax_data -> compute_pkgs)      = NULL;
   (relax_data -> comm_handle)       = NULL;

   *relax_vdata_ptr = (void *) relax_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxDestroy( void *relax_vdata )
{
   hypre_SSAMGRelaxData  *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;
   hypre_SStructPMatrix  *pmatrix;
   HYPRE_Int              part, nparts, nvars, vi, i;

   if (relax_data)
   {
      nparts = (relax_data -> nparts);
      for (part = 0; part < nparts; part++)
      {
         pmatrix = hypre_SStructMatrixPMatrix((relax_data -> A), part);
         nvars   = hypre_SStructPMatrixNVars(pmatrix);
         for (i = 0; i < (relax_data -> num_nodesets); i++)
         {
            hypre_TFree(relax_data -> nodeset_indices[i], HYPRE_MEMORY_HOST);
            for (vi = 0; vi < nvars; vi++)
            {
               hypre_ComputePkgDestroy(relax_data -> svec_compute_pkgs[part][i][vi]);
            }
            hypre_TFree(relax_data -> svec_compute_pkgs[part][i], HYPRE_MEMORY_HOST);
            hypre_ComputePkgDestroy(relax_data -> compute_pkgs[part][i]);
         }
         hypre_TFree(relax_data -> svec_compute_pkgs[part], HYPRE_MEMORY_HOST);
         hypre_TFree(relax_data -> comm_handle[part], HYPRE_MEMORY_HOST);
         hypre_TFree(relax_data -> compute_pkgs[part], HYPRE_MEMORY_HOST);
      }
      HYPRE_SStructMatrixDestroy(relax_data -> A);
      HYPRE_SStructVectorDestroy(relax_data -> b);
      HYPRE_SStructVectorDestroy(relax_data -> x);
      HYPRE_SStructVectorDestroy(relax_data -> t);

      hypre_TFree(relax_data -> nodeset_sizes, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> nodeset_ranks, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> nodeset_strides, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> nodeset_indices, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> svec_compute_pkgs, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> comm_handle, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> compute_pkgs, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> x_loc, HYPRE_MEMORY_HOST);
      hypre_TFree((relax_data ->A_loc)[0], HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> A_loc, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> bp, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> xp, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> tp, HYPRE_MEMORY_HOST);
      for (vi = 0; vi < nvars; vi++)
      {
         hypre_TFree((relax_data -> Ap)[vi], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(relax_data -> Ap, HYPRE_MEMORY_HOST);
      hypre_TFree(relax_data -> mweights, HYPRE_MEMORY_HOST);
      hypre_FinalizeTiming(relax_data -> time_index);
      hypre_TFree(relax_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetTol( void       *relax_vdata,
                        HYPRE_Real  tol )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   (relax_data -> tol) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetMaxIter( void       *relax_vdata,
                            HYPRE_Int   max_iter )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   (relax_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetZeroGuess( void      *relax_vdata,
                              HYPRE_Int  zero_guess  )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   (relax_data -> zero_guess) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetWeights( void        *relax_vdata,
                            HYPRE_Real  *weights )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   (relax_data -> weights) = weights;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetActiveParts( void       *relax_vdata,
                                HYPRE_Int  *active_p )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   (relax_data -> active_p) = active_p;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetMatvecData( void  *relax_vdata,
                               void  *matvec_vdata )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   (relax_data -> matvec_vdata) = matvec_vdata;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetNumNodesets( void      *relax_vdata,
                                HYPRE_Int  num_nodesets )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int             i;

   /* free up old nodeset memory */
   for (i = 0; i < (relax_data -> num_nodesets); i++)
   {
      hypre_TFree(relax_data -> nodeset_indices[i], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(relax_data -> nodeset_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> nodeset_ranks, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> nodeset_strides, HYPRE_MEMORY_HOST);
   hypre_TFree(relax_data -> nodeset_indices, HYPRE_MEMORY_HOST);

   /* alloc new nodeset memory */
   (relax_data -> num_nodesets)    = num_nodesets;
   (relax_data -> nodeset_sizes)   = hypre_TAlloc(HYPRE_Int    , num_nodesets, HYPRE_MEMORY_HOST);
   (relax_data -> nodeset_ranks)   = hypre_TAlloc(HYPRE_Int    , num_nodesets, HYPRE_MEMORY_HOST);
   (relax_data -> nodeset_strides) = hypre_TAlloc(hypre_Index  , num_nodesets, HYPRE_MEMORY_HOST);
   (relax_data -> nodeset_indices) = hypre_TAlloc(hypre_Index *, num_nodesets, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_nodesets; i++)
   {
      (relax_data -> nodeset_sizes[i])   = 0;
      (relax_data -> nodeset_ranks[i])   = i;
      (relax_data -> nodeset_indices[i]) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetNodeset( void         *relax_vdata,
                            HYPRE_Int     nodeset,
                            HYPRE_Int     nodeset_size,
                            hypre_Index   nodeset_stride,
                            hypre_Index  *nodeset_indices )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int             i;

   /* free up old nodeset memory */
   hypre_TFree(relax_data -> nodeset_indices[nodeset], HYPRE_MEMORY_HOST);

   /* alloc new nodeset memory */
   (relax_data -> nodeset_sizes[nodeset])   = nodeset_size;
   (relax_data -> nodeset_indices[nodeset]) = hypre_TAlloc(hypre_Index, nodeset_size,
                                                           HYPRE_MEMORY_HOST);

   /* set values */
   hypre_CopyIndex(nodeset_stride, (relax_data -> nodeset_strides[nodeset]));
   for (i = 0; i < nodeset_size; i++)
   {
      hypre_CopyIndex(nodeset_indices[i], (relax_data -> nodeset_indices[nodeset][i]));
   }

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetNodesetRank( void *relax_vdata,
                                HYPRE_Int   nodeset,
                                HYPRE_Int   nodeset_rank )
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   (relax_data -> nodeset_ranks[nodeset]) = nodeset_rank;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetType( void      *relax_vdata,
                         HYPRE_Int  type)
{
   hypre_SSAMGRelaxData *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;
   hypre_Index           stride;

   (relax_data -> type) = type;
   switch (type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
      {
         hypre_Index  indices[1];

         hypre_SetIndex(stride, 1);
         hypre_SetIndex(indices[0], 0);
         hypre_SSAMGRelaxSetNumNodesets(relax_data, 1);
         hypre_SSAMGRelaxSetNodeset(relax_vdata, 0, 1, stride, indices);
      }
      break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         /* TODO: extend this to 3D */

         hypre_Index red[4], black[4];

         hypre_SetIndex(stride, 2);

         /* define red points (point set 0) */
         hypre_SetIndex3(red[0], 1, 0, 0);
         hypre_SetIndex3(red[1], 0, 1, 0);
         hypre_SetIndex3(red[2], 0, 0, 1);
         hypre_SetIndex3(red[3], 1, 1, 1);

         /* define black points (point set 1) */
         hypre_SetIndex3(black[0], 0, 0, 0);
         hypre_SetIndex3(black[1], 1, 1, 0);
         hypre_SetIndex3(black[2], 1, 0, 1);
         hypre_SetIndex3(black[3], 0, 1, 1);

         hypre_SSAMGRelaxSetNumNodesets(relax_data, 2);
         hypre_SSAMGRelaxSetNodeset(relax_data, 0, 4, stride, red);
         hypre_SSAMGRelaxSetNodeset(relax_data, 1, 4, stride, black);
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetPreRelax( void  *relax_vdata )
{
   hypre_SSAMGRelaxData  *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int              type       = (relax_data -> type);

   switch (type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         hypre_SSAMGRelaxSetNodesetRank(relax_data, 0, 0);
         hypre_SSAMGRelaxSetNodesetRank(relax_data, 1, 1);
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetPostRelax( void  *relax_vdata )
{
   hypre_SSAMGRelaxData  *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int              type       = (relax_data -> type);

   switch (type)
   {
      case 1: /* Weighted Jacobi */
      case 0: /* Jacobi */
         break;

      case 2: /* Red-Black Gauss-Seidel */
      {
         hypre_SSAMGRelaxSetNodesetRank(relax_data, 0, 1);
         hypre_SSAMGRelaxSetNodesetRank(relax_data, 1, 0);
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetTempVec( void                *relax_vdata,
                            hypre_SStructVector *t )
{
   hypre_SSAMGRelaxData   *relax_data = (hypre_SSAMGRelaxData *) relax_vdata;

   HYPRE_SStructVectorDestroy(relax_data -> t);
   hypre_SStructVectorRef(t, &(relax_data -> t));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxSetup( void                *relax_vdata,
                       hypre_SStructMatrix *A,
                       hypre_SStructVector *b,
                       hypre_SStructVector *x )
{
   hypre_SSAMGRelaxData   *relax_data      = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int               num_nodesets    = (relax_data -> num_nodesets);
   HYPRE_Int              *nodeset_sizes   = (relax_data -> nodeset_sizes);
   hypre_Index            *nodeset_strides = (relax_data -> nodeset_strides);
   hypre_Index           **nodeset_indices = (relax_data -> nodeset_indices);
   HYPRE_Real             *weights         = (relax_data -> weights);
   HYPRE_Int               ndim            = hypre_SStructMatrixNDim(A);
   HYPRE_Int               nparts          = hypre_SStructMatrixNParts(A);

   MPI_Comm                comm;
   hypre_SStructGrid      *grid;
   hypre_StructGrid       *sgrid;
   hypre_SStructVector    *t;
   hypre_SStructPVector   *px;
   hypre_StructVector     *sx;
   hypre_SStructPMatrix   *pA;
   hypre_StructMatrix     *sA;

   hypre_ComputePkg    ****svec_compute_pkgs;
   hypre_CommHandle     ***comm_handle;
   hypre_ComputePkg     ***compute_pkgs;
   hypre_ComputeInfo      *compute_info;

   HYPRE_Real            **bp;
   HYPRE_Real            **xp;
   HYPRE_Real            **tp;
   HYPRE_Real           ***Ap;
   HYPRE_Real             *x_loc;
   HYPRE_Real            **A_loc;
   HYPRE_Real             *mweights;

   hypre_StructStencil    *sstencil;
   hypre_Index            *sstencil_shape;
   HYPRE_Int               sstencil_size;
   hypre_StructStencil    *sstencil_union;
   hypre_Index            *sstencil_union_shape;
   HYPRE_Int               sstencil_union_count;

   hypre_IndexRef          stride;
   hypre_IndexRef          index;
   hypre_BoxArrayArray    *orig_indt_boxes;
   hypre_BoxArrayArray    *orig_dept_boxes;
   hypre_BoxArrayArray    *box_aa;
   hypre_BoxArray         *box_a;
   hypre_Box              *box;
   HYPRE_Int               box_aa_size;
   HYPRE_Int               box_a_size;
   hypre_BoxArrayArray    *new_box_aa;
   hypre_BoxArray         *new_box_a;
   hypre_Box              *new_box;

   HYPRE_Int               compute_i;
   HYPRE_Int               i, j, k, m, s, vi, vj, part;
   HYPRE_Int               nvars;
   HYPRE_Int               set;

   /*----------------------------------------------------------
    * Set up the temp vector
    *----------------------------------------------------------*/
   if ((relax_data -> t) == NULL)
   {
      comm = hypre_SStructVectorComm(b);
      grid = hypre_SStructVectorGrid(b);

      HYPRE_SStructVectorCreate(comm, grid, &t);
      HYPRE_SStructVectorInitialize(t);
      HYPRE_SStructVectorAssemble(t);
      (relax_data -> t) = t;
   }

   /*----------------------------------------------------------
    * Set mweights = (1 - weights)
    *----------------------------------------------------------*/
   mweights = hypre_TAlloc(HYPRE_Real, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      mweights[part] = 1.0 - weights[part];
   }
   (relax_data -> mweights) = mweights;

   /*----------------------------------------------------------
    * Allocate storage used to invert local diagonal blocks
    *----------------------------------------------------------*/
   // TODO: Why nthreads is used in all TAllocs?
   nvars = 1;
   x_loc = hypre_TAlloc(HYPRE_Real  , hypre_NumThreads()*nvars, HYPRE_MEMORY_HOST);
   A_loc = hypre_TAlloc(HYPRE_Real *, hypre_NumThreads()*nvars, HYPRE_MEMORY_HOST);

   // nvars*nvars is probably not needed here!
   A_loc[0] = hypre_TAlloc(HYPRE_Real  , hypre_NumThreads()*nvars*nvars, HYPRE_MEMORY_HOST);

   for (vi = 1; vi < hypre_NumThreads()*nvars; vi++)
   {
      A_loc[vi] = A_loc[0] + vi*nvars;
   }

   /* Allocate pointers for vector and matrix */
   bp = hypre_TAlloc(HYPRE_Real  *, nvars, HYPRE_MEMORY_HOST);
   xp = hypre_TAlloc(HYPRE_Real  *, nvars, HYPRE_MEMORY_HOST);
   tp = hypre_TAlloc(HYPRE_Real  *, nvars, HYPRE_MEMORY_HOST);
   Ap = hypre_TAlloc(HYPRE_Real **, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      Ap[vi] = hypre_TAlloc(HYPRE_Real *, nvars, HYPRE_MEMORY_HOST);
   }

   /*----------------------------------------------------------
    * Set up the compute packages for each part
    *----------------------------------------------------------*/
   svec_compute_pkgs = hypre_CTAlloc(hypre_ComputePkg ***, nparts, HYPRE_MEMORY_HOST);
   compute_pkgs      = hypre_CTAlloc(hypre_ComputePkg  **, nparts, HYPRE_MEMORY_HOST);
   comm_handle       = hypre_CTAlloc(hypre_CommHandle  **, nparts, HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      nvars = hypre_SStructPMatrixNVars(pA);
      if (nvars > 0)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, 0, 0);
         sgrid = hypre_StructMatrixGrid(sA);
      }

      svec_compute_pkgs[part] = hypre_CTAlloc(hypre_ComputePkg **, num_nodesets,
                                              HYPRE_MEMORY_HOST);
      compute_pkgs[part]      = hypre_CTAlloc(hypre_ComputePkg  *, num_nodesets,
                                              HYPRE_MEMORY_HOST);
      comm_handle[part]       = hypre_CTAlloc(hypre_CommHandle  *, nvars,
                                              HYPRE_MEMORY_HOST);

      for (set = 0; set < num_nodesets; set++)
      {
         /*----------------------------------------------------------
          * Set up the compute packages to define sends and receives
          * for each StructVector (svec_compute_pkgs) and the compute
          * package to define independent and dependent computations
          * (compute_pkgs).
          *----------------------------------------------------------*/
         svec_compute_pkgs[part][set] = hypre_CTAlloc(hypre_ComputePkg *, nvars,
                                                      HYPRE_MEMORY_HOST);

         /*----------------------------------------------------------
          * The first execution (vi=-1) sets up the stencil to
          * define independent and dependent computations. The
          * stencil is the "union" over i,j of all stencils for
          * for struct_matrix A_ij.
          *
          * Other executions (vi > -1) set up the stencil to
          * define sends and recieves for the struct_vector vi.
          * The stencil for vector i is the "union" over j of all
          * stencils for struct_matrix A_ji.
          *----------------------------------------------------------*/
         for (vi = -1; vi < nvars; vi++)
         {
            sstencil_union_count = 0;
            if (vi == -1)
            {
               for (i = 0; i < nvars; i++)
               {
                  for (vj = 0; vj < nvars; vj++)
                  {
                     if (hypre_SStructPMatrixSMatrix(pA,vj,i) != NULL)
                     {
                        sstencil = hypre_SStructPMatrixSStencil(pA, vj, i);
                        sstencil_union_count +=
                           hypre_StructStencilSize(sstencil);
                     }
                  }
               }
            }
            else
            {
               for (vj = 0; vj < nvars; vj++)
               {
                  if (hypre_SStructPMatrixSMatrix(pA,vj,vi) != NULL)
                  {
                     sstencil = hypre_SStructPMatrixSStencil(pA, vj, vi);
                     sstencil_union_count += hypre_StructStencilSize(sstencil);
                  }
               }
            }
            sstencil_union_shape = hypre_CTAlloc(hypre_Index, sstencil_union_count,
                                                 HYPRE_MEMORY_HOST);
            sstencil_union_count = 0;
            if (vi == -1)
            {
               for (i = 0; i < nvars; i++)
               {
                  for (vj = 0; vj < nvars; vj++)
                  {
                     if (hypre_SStructPMatrixSMatrix(pA,vj,i) != NULL)
                     {
                        sstencil = hypre_SStructPMatrixSStencil(pA, vj, i);
                        sstencil_size = hypre_StructStencilSize(sstencil);
                        sstencil_shape = hypre_StructStencilShape(sstencil);
                        for (s = 0; s < sstencil_size; s++)
                        {
                           hypre_CopyIndex(sstencil_shape[s],
                                           sstencil_union_shape[sstencil_union_count]);
                           sstencil_union_count++;
                        }
                     }
                  }
               }
            }
            else
            {
               for (vj = 0; vj < nvars; vj++)
               {
                  if (hypre_SStructPMatrixSMatrix(pA,vj,vi) != NULL)
                  {
                     sstencil = hypre_SStructPMatrixSStencil(pA, vj, vi);
                     sstencil_size = hypre_StructStencilSize(sstencil);
                     sstencil_shape = hypre_StructStencilShape(sstencil);
                     for (s = 0; s < sstencil_size; s++)
                     {
                        hypre_CopyIndex(sstencil_shape[s],
                                        sstencil_union_shape[sstencil_union_count]);
                        sstencil_union_count++;
                     }
                  }
               }
            }

            sstencil_union = hypre_StructStencilCreate(ndim,
                                                       sstencil_union_count,
                                                       sstencil_union_shape);
            hypre_CreateComputeInfo(sgrid, sstencil_union, &compute_info);
            orig_indt_boxes = hypre_ComputeInfoIndtBoxes(compute_info);
            orig_dept_boxes = hypre_ComputeInfoDeptBoxes(compute_info);
            stride = nodeset_strides[set];

            for (compute_i = 0; compute_i < 2; compute_i++)
            {
               if (compute_i)
               {
                  box_aa = orig_dept_boxes;
               }
               else
               {
                  box_aa = orig_indt_boxes;
               }

               box_aa_size = hypre_BoxArrayArraySize(box_aa);
               new_box_aa = hypre_BoxArrayArrayCreate(box_aa_size, ndim);

               for (i = 0; i < box_aa_size; i++)
               {
                  box_a = hypre_BoxArrayArrayBoxArray(box_aa, i);
                  box_a_size = hypre_BoxArraySize(box_a);
                  new_box_a = hypre_BoxArrayArrayBoxArray(new_box_aa, i);
                  hypre_BoxArraySetSize(new_box_a, box_a_size * nodeset_sizes[set]);

                  k = 0;
                  for (m = 0; m < nodeset_sizes[set]; m++)
                  {
                     index = nodeset_indices[set][m];

                     for (j = 0; j < box_a_size; j++)
                     {
                        box = hypre_BoxArrayBox(box_a, j);
                        new_box = hypre_BoxArrayBox(new_box_a, k);

                        hypre_CopyBox(box, new_box);
                        hypre_ProjectBox(new_box, index, stride);

                        k++;
                     }
                  }
               }

               if (compute_i)
               {
                  hypre_ComputeInfoDeptBoxes(compute_info) = new_box_aa;
               }
               else
               {
                  hypre_ComputeInfoIndtBoxes(compute_info) = new_box_aa;
               }
            }

            hypre_CopyIndex(stride, hypre_ComputeInfoStride(compute_info));

            if (nvars > 0)
            {
               if (vi == -1)
               {
                  sx = hypre_SStructPVectorSVector(px, 0);
                  hypre_ComputePkgCreate(compute_info,
                                         hypre_StructVectorDataSpace(sx),
                                         1, sgrid, &compute_pkgs[part][set]);
               }
               else
               {
                  sx = hypre_SStructPVectorSVector(px, vi);
                  hypre_ComputePkgCreate(compute_info,
                                         hypre_StructVectorDataSpace(sx),
                                         1, sgrid, &svec_compute_pkgs[part][set][vi]);
               }
            }

            hypre_BoxArrayArrayDestroy(orig_indt_boxes);
            hypre_BoxArrayArrayDestroy(orig_dept_boxes);
            hypre_StructStencilDestroy(sstencil_union);
         } /* loop on nvars */
      } /* loop on nodesets */
   } /* loop on parts */

   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/
   hypre_SStructMatrixRef(A, &(relax_data -> A));
   hypre_SStructVectorRef(x, &(relax_data -> x));
   hypre_SStructVectorRef(b, &(relax_data -> b));

   (relax_data -> A_loc)             = A_loc;
   (relax_data -> x_loc)             = x_loc;
   (relax_data -> Ap)                = Ap;
   (relax_data -> bp)                = bp;
   (relax_data -> tp)                = tp;
   (relax_data -> xp)                = xp;
   (relax_data -> svec_compute_pkgs) = svec_compute_pkgs;
   (relax_data -> compute_pkgs)      = compute_pkgs;
   (relax_data -> comm_handle)       = comm_handle;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelax( void                *relax_vdata,
                  hypre_SStructMatrix *A,
                  hypre_SStructVector *b,
                  hypre_SStructVector *x )
{
   hypre_SSAMGRelaxData  *relax_data   = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int              num_nodesets = (relax_data -> num_nodesets);
   HYPRE_Int              zero_guess   = (relax_data -> zero_guess);
   HYPRE_Int             *active_p     = (relax_data -> active_p);
   HYPRE_Int              nparts       = hypre_SStructMatrixNParts(A);

   hypre_SStructPVector  *px;
   HYPRE_Int              part;

   if (num_nodesets == 1)
   {
      hypre_SSAMGRelaxMV(relax_vdata, A, b, x);
   }
   else
   {
      hypre_SSAMGRelaxGeneric(relax_vdata, A, b, x);
   }

   /* Set x=0, so r=(b-Ax)=b on inactive parts */
   if (zero_guess)
   {
      for (part = 0; part < nparts; part++)
      {
         if (!active_p[part])
         {
            px = hypre_SStructVectorPVector(x, part);
            hypre_SStructPVectorSetConstantValues(px, 0.0);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SSAMGRelaxGeneric
 *
 * Computes x_{k+1} = (1 - w)*x_k + w*inv(D)*(b - (L + U)*x_k)
 * Does not unroll stencil loops. Use hypre_SSAMGRelaxMV for better performance
 *
 * TODO:
 *       1) Do we really need nodesets?
 *       2) Can we reduce communication?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxGeneric( void                *relax_vdata,
                         hypre_SStructMatrix *A,
                         hypre_SStructVector *b,
                         hypre_SStructVector *x )
{
   hypre_SSAMGRelaxData    *relax_data        = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int                max_iter          = (relax_data -> max_iter);
   HYPRE_Int                zero_guess        = (relax_data -> zero_guess);
   HYPRE_Int               *active_p          = (relax_data -> active_p);
   HYPRE_Real              *weights           = (relax_data -> weights);
   HYPRE_Int                num_nodesets      = (relax_data -> num_nodesets);
   HYPRE_Int               *nodeset_ranks     = (relax_data -> nodeset_ranks);
   hypre_Index             *nodeset_strides   = (relax_data -> nodeset_strides);
   hypre_ComputePkg      ***compute_pkgs      = (relax_data -> compute_pkgs);
   hypre_ComputePkg     ****svec_compute_pkgs = (relax_data -> svec_compute_pkgs);
   hypre_CommHandle      ***comm_handle       = (relax_data -> comm_handle);
   hypre_SStructVector     *t                 = (relax_data -> t);
   HYPRE_Int                ndim              = hypre_SStructMatrixNDim(A);
   HYPRE_Int                nparts            = hypre_SStructMatrixNParts(A);
   hypre_ParCSRMatrix      *uA                = hypre_SStructMatrixParCSRMatrix(A);

   hypre_StructMatrix      *sA;
   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px, *pb, *pt;
   hypre_StructVector      *sx, *sb, *st;
   hypre_ParVector         *ux, *ut;

   HYPRE_Real              *Ap;
   HYPRE_Real              *bp;
   HYPRE_Real              *xp;
   HYPRE_Real              *tp;

   hypre_StructStencil     *stencil;
   HYPRE_Int                stencil_diag;
   HYPRE_Int                stencil_size;
   hypre_Index             *stencil_shape;

   hypre_ComputePkg        *compute_pkg;
   hypre_ComputePkg        *svec_compute_pkg;
   hypre_BoxArrayArray     *compute_box_aa;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;
   hypre_Box               *A_data_box;
   hypre_Box               *b_data_box;
   hypre_Box               *x_data_box;
   hypre_Box               *t_data_box;

   hypre_IndexRef           stride;
   hypre_IndexRef           start;
   hypre_Index              loop_size;

   HYPRE_Int                iter;
   HYPRE_Int                part, nvars, set, nodeset;
   HYPRE_Int                offset;
   HYPRE_Int                compute_i, i, j, vi, vj, si;
   HYPRE_Complex            zero = 0.0;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_BeginTiming(relax_data -> time_index);

   HYPRE_SStructMatrixDestroy(relax_data -> A);
   HYPRE_SStructVectorDestroy(relax_data -> b);
   HYPRE_SStructVectorDestroy(relax_data -> x);
   hypre_SStructMatrixRef(A, &(relax_data -> A));
   hypre_SStructVectorRef(x, &(relax_data -> x));
   hypre_SStructVectorRef(b, &(relax_data -> b));

   (relax_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SStructVectorSetConstantValues(x, zero);
      }

      hypre_EndTiming(relax_data -> time_index);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   iter = 0;
   if (zero_guess)
   {
      if (num_nodesets > 1)
      {
         hypre_SStructVectorSetConstantValues(x, zero);
      }

      for (part = 0; part < nparts; part++)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s %d", "Diag scale part", part);

         if (active_p[part])
         {
            pA    = hypre_SStructMatrixPMatrix(A, part);
            px    = hypre_SStructVectorPVector(x, part);
            pb    = hypre_SStructVectorPVector(b, part);
            nvars = hypre_SStructPMatrixNVars(pA);

            for (set = 0; set < num_nodesets; set++)
            {
               nodeset     = nodeset_ranks[set];
               stride      = nodeset_strides[nodeset];
               compute_pkg = compute_pkgs[part][nodeset];

               for (compute_i = 0; compute_i < 2; compute_i++)
               {
                  if (compute_i)
                  {
                     compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
                  }
                  else
                  {
                     compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
                  }

                  hypre_ForBoxArrayI(i, compute_box_aa)
                  {
                     compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);
                     for (vi = 0; vi < nvars; vi++)
                     {
                        sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
                        sb = hypre_SStructPVectorSVector(pb, vi);
                        sx = hypre_SStructPVectorSVector(px, vi);

                        A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(sA), i);
                        b_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(sb), i);
                        x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(sx), i);

                        // Implement hypre_StructMatrixDiagData ?
                        stencil = hypre_StructMatrixStencil(sA);
                        stencil_diag = hypre_StructStencilDiagEntry(stencil);
                        Ap = hypre_StructMatrixBoxData(sA, i, stencil_diag);

                        bp = hypre_StructVectorBoxData(sb, i);
                        xp = hypre_StructVectorBoxData(sx, i);

                        hypre_ForBoxI(j, compute_box_a)
                        {
                           compute_box = hypre_BoxArrayBox(compute_box_a, j);

                           start = hypre_BoxIMin(compute_box);
                           hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                           if (weights[part] != 1.0)
                           {
                              hypre_BoxLoop3Begin(ndim, loop_size,
                                                  A_data_box, start, stride, Ai,
                                                  x_data_box, start, stride, xi,
                                                  b_data_box, start, stride, bi);
                              {
                                 xp[xi] = weights[part] * bp[bi] / Ap[Ai];
                              }
                              hypre_BoxLoop3End(Ai, xi, bi);
                           }
                           else
                           {
                              hypre_BoxLoop3Begin(ndim, loop_size,
                                                  A_data_box, start, stride, Ai,
                                                  x_data_box, start, stride, xi,
                                                  b_data_box, start, stride, bi);
                              {
                                 xp[xi] = bp[bi] / Ap[Ai];
                              }
                              hypre_BoxLoop3End(Ai, xi, bi);
                           }
                        } /* hypre_ForBoxI */
                     } /* loop on vars */
                  } /* hypre_ForBoxArrayI */
               } /* loop on compute_i */
            } /* loop on sets */
         } /* if (active_p[part])  */
         HYPRE_ANNOTATE_REGION_END("%s %d", "Diag scale part", part);
      } /* loop on parts */

      iter++;
   } /* if (zero_guess) */

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/
   for (; iter < max_iter; iter++)
   {
      for (part = 0; part < nparts; part++)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s %d", "Residual part", part);

         pt = hypre_SStructVectorPVector(t, part);
         if (active_p[part])
         {
            pA = hypre_SStructMatrixPMatrix(A, part);
            px = hypre_SStructVectorPVector(x, part);
            pb = hypre_SStructVectorPVector(b, part);
            nvars = hypre_SStructPMatrixNVars(pA);

            for (set = 0; set < num_nodesets; set++)
            {
               nodeset = nodeset_ranks[set];
               stride  = nodeset_strides[nodeset];
               compute_pkg = compute_pkgs[part][nodeset];

               for (compute_i = 0; compute_i < 2; compute_i++)
               {
                  switch(compute_i)
                  {
                     case 0:
                     {
                        for (vi = 0; vi < nvars; vi++)
                        {
                           sx = hypre_SStructPVectorSVector(px, vi);
                           xp = hypre_StructVectorData(sx);
                           svec_compute_pkg = svec_compute_pkgs[part][nodeset][vi];
                           hypre_InitializeIndtComputations(svec_compute_pkg,
                                                            xp, &comm_handle[part][vi]);
                        }
                        compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
                     }
                     break;

                     case 1:
                     {
                        for (vi = 0; vi < nvars; vi++)
                        {
                           hypre_FinalizeIndtComputations(comm_handle[part][vi]);
                        }
                        compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
                     }
                     break;
                  }

                  hypre_ForBoxArrayI(i, compute_box_aa)
                  {
                     compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

                     sA = hypre_SStructPMatrixSMatrix(pA, 0, 0);
                     sb = hypre_SStructPVectorSVector(pb, 0);
                     sx = hypre_SStructPVectorSVector(px, 0);
                     st = hypre_SStructPVectorSVector(pt, 0);

                     A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(sA), i);
                     b_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(sb), i);
                     x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(sx), i);
                     t_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(st), i);

                     hypre_ForBoxI(j, compute_box_a)
                     {
                        compute_box = hypre_BoxArrayBox(compute_box_a, j);
                        start  = hypre_BoxIMin(compute_box);
                        hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                        for (vi = 0; vi < nvars; vi++)
                        {
                           sb = hypre_SStructPVectorSVector(pb, vi);
                           st = hypre_SStructPVectorSVector(pt, vi);
                           bp = hypre_StructVectorBoxData(sb, i);
                           tp = hypre_StructVectorBoxData(st, i);

                           /* Copy rhs into temp vector */
                           hypre_BoxLoop2Begin(ndim, loop_size,
                                               b_data_box, start, stride, bi,
                                               t_data_box, start, stride, ti);
                           {
                              tp[ti] = bp[bi];
                           }
                           hypre_BoxLoop2End(bi, ti);

                           for (vj = 0; vj < nvars; vj++)
                           {
                              sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
                              if (sA != NULL)
                              {
                                 sx = hypre_SStructPVectorSVector(px, vj);
                                 stencil = hypre_StructMatrixStencil(sA);
                                 stencil_shape = hypre_StructStencilShape(stencil);
                                 stencil_size  = hypre_StructStencilSize(stencil);
                                 stencil_diag  = hypre_StructStencilDiagEntry(stencil);

                                 for (si = 0; si < stencil_size; si++)
                                 {
                                    if (si != stencil_diag)
                                    {
                                       offset = hypre_BoxOffsetDistance(x_data_box,
                                                                        stencil_shape[si]);
                                       Ap = hypre_StructMatrixBoxData(sA, i, si);
                                       xp = hypre_StructVectorBoxData(sx, i) + offset;

                                       hypre_BoxLoop3Begin(ndim, loop_size,
                                                           A_data_box, start, stride, Ai,
                                                           x_data_box, start, stride, xi,
                                                           t_data_box, start, stride, ti);
                                       {
                                          tp[ti] -= Ap[Ai] * xp[xi];
                                       }
                                       hypre_BoxLoop3End(Ai, xi, ti);
                                    }
                                 } /* loop on stencil entries */
                              } /* if (sA != NULL) */
                           } /* loop on j-vars */
                        } /* loop on i-vars */
                     } /* hypre_ForBoxI */
                  } /* hypre_ForBoxArrayI */
               } /* loop on compute_i */
            } /* loop on sets */
         } /* if (active_p[part]) */
         HYPRE_ANNOTATE_REGION_END("%s %d", "Residual part", part);
      } /* loop on parts */

      /* Compute unstructured component: t = t - U*x */
      hypre_SStructVectorConvert(x, &ux);
      hypre_SStructVectorConvert(t, &ut);
      hypre_ParCSRMatrixMatvec(-1.0, uA, ux, 1.0, ut);
      hypre_SStructVectorRestore(x, NULL);
      hypre_SStructVectorRestore(t, ut);

      /* Apply diagonal scaling */
      for (part = 0; part < nparts; part++)
      {
         HYPRE_ANNOTATE_REGION_BEGIN("%s %d", "Diag scale part", part);

         px = hypre_SStructVectorPVector(x, part);
         if (active_p[part])
         {
            pA = hypre_SStructMatrixPMatrix(A, part);
            pt = hypre_SStructVectorPVector(t, part);
            nvars = hypre_SStructPMatrixNVars(pA);
            for (vi = 0; vi < nvars; vi++)
            {
               sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
               sx = hypre_SStructPVectorSVector(px, vi);
               st = hypre_SStructPVectorSVector(pt, vi);

               stencil       = hypre_StructMatrixStencil(sA);
               stencil_diag  = hypre_StructStencilDiagEntry(stencil);
               compute_box_a = hypre_StructGridBoxes(hypre_StructMatrixGrid(sA));
               hypre_ForBoxI(i, compute_box_a)
               {
                  compute_box = hypre_BoxArrayBox(compute_box_a, i);
                  start  = hypre_BoxIMin(compute_box);
                  hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                  A_data_box =  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(sA), i);
                  x_data_box =  hypre_BoxArrayBox(hypre_StructVectorDataSpace(sx), i);
                  t_data_box =  hypre_BoxArrayBox(hypre_StructVectorDataSpace(st), i);

                  Ap = hypre_StructMatrixBoxData(sA, i, stencil_diag);
                  xp = hypre_StructVectorBoxData(sx, i);
                  tp = hypre_StructVectorBoxData(st, i);

                  if (weights[part] != 1.0)
                  {
                     hypre_BoxLoop3Begin(ndim, loop_size,
                                         A_data_box, start, stride, Ai,
                                         x_data_box, start, stride, xi,
                                         t_data_box, start, stride, ti);
                     {
                        xp[xi] = (1.0 - weights[part]) * xp[xi] +
                                  weights[part] * tp[ti] / Ap[Ai];
                     }
                     hypre_BoxLoop3End(Ai, xi, ti);
                  }
                  else
                  {
                     hypre_BoxLoop3Begin(ndim, loop_size,
                                         A_data_box, start, stride, Ai,
                                         x_data_box, start, stride, xi,
                                         t_data_box, start, stride, ti);
                     {
                        xp[xi] = tp[ti] / Ap[Ai];
                     }
                     hypre_BoxLoop3End(Ai, xi, ti);
                  } /* if (weights[part] != 1.0) */
               } /* hypre_ForBoxI(i, compute_box_a) */
            } /* loop on vars */
         }
         HYPRE_ANNOTATE_REGION_END("%s %d", "Diag scale part", part);
      } /* loop on parts */
   } /* loop on iterations */

   (relax_data -> num_iterations) = iter;
   hypre_EndTiming(relax_data -> time_index);
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGRelaxMV( void                *relax_vdata,
                    hypre_SStructMatrix *A,
                    hypre_SStructVector *b,
                    hypre_SStructVector *x )
{
   hypre_SSAMGRelaxData    *relax_data        = (hypre_SSAMGRelaxData *) relax_vdata;
   HYPRE_Int                max_iter          = (relax_data -> max_iter);
   HYPRE_Int                zero_guess        = (relax_data -> zero_guess);
   HYPRE_Real              *weights           = (relax_data -> weights);
   HYPRE_Real              *mweights          = (relax_data -> mweights);
   HYPRE_Int                num_nodesets      = (relax_data -> num_nodesets);
   void                    *matvec_vdata      = (relax_data -> matvec_vdata);
   hypre_SStructVector     *t                 = (relax_data -> t);

   HYPRE_Int                iter = 0;
   HYPRE_Complex            zero = 0.0;
   HYPRE_Complex            one  = 1.0;
   HYPRE_Complex            mone = -1.0;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_BeginTiming(relax_data -> time_index);

   (relax_data -> num_iterations) = 0;
   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SStructVectorSetConstantValues(x, zero);
      }

      hypre_EndTiming(relax_data -> time_index);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   if (num_nodesets > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "num_nodesets > 1 not supported!");
      hypre_EndTiming(relax_data -> time_index);
      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/
   if (zero_guess)
   {
      /* x = w*inv(D)*b */
      hypre_SStructMatrixInvDiagAxpy(matvec_vdata, weights, A, b, NULL, x);
      iter++;
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/
   for (; iter < max_iter; iter++)
   {
      /* t = b - (L + U)*x */
      hypre_SStructMatvecSetSkipDiag(matvec_vdata, 1);
      hypre_SStructMatvecCompute(matvec_vdata, mone, A, x, one, b, t);
      hypre_SStructMatvecSetSkipDiag(matvec_vdata, 0);

      /* x = (1 - w)*x + w*inv(D)*t */
      hypre_SStructMatrixInvDiagAxpy(matvec_vdata, weights, A, t, mweights, x);
   }

   (relax_data -> num_iterations) = iter;
   hypre_EndTiming(relax_data -> time_index);
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
