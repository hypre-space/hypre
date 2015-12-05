/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.18 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
                       
   double                  tol;                /* not yet used */
   HYPRE_Int               max_iter;
   HYPRE_Int               rel_change;         /* not yet used */
   HYPRE_Int               zero_guess;
   double                  weight;
                         
   HYPRE_Int               num_nodesets;
   HYPRE_Int              *nodeset_sizes;
   HYPRE_Int              *nodeset_ranks;
   hypre_Index            *nodeset_strides;
   hypre_Index           **nodeset_indices;
                       
   hypre_SStructPMatrix   *A;
   hypre_SStructPVector   *b;
   hypre_SStructPVector   *x;

   hypre_SStructPVector   *t;

   HYPRE_Int             **diag_rank;

   /* defines sends and recieves for each struct_vector */
   hypre_ComputePkg     ***svec_compute_pkgs;
   hypre_CommHandle     **comm_handle;

   /* defines independent and dependent boxes for computations */
   hypre_ComputePkg      **compute_pkgs;

   /* pointers to local storage used to invert diagonal blocks */
   double               **A_loc;
   double                *x_loc;

   /* pointers for vector and matrix data */	
   double              ***Ap;
   double               **bp;
   double               **xp;
   double               **tp;


   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;
   HYPRE_Int               flops;

} hypre_NodeRelaxData;

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxCreate
 *--------------------------------------------------------------------------*/

void *
hypre_NodeRelaxCreate( MPI_Comm  comm )
{
   hypre_NodeRelaxData *relax_data;

   hypre_Index           stride;
   hypre_Index           indices[1];

   relax_data = hypre_CTAlloc(hypre_NodeRelaxData, 1);

   (relax_data -> comm)       = comm;
   (relax_data -> time_index) = hypre_InitializeTiming("NodeRelax");

   /* set defaults */
   (relax_data -> tol)              = 1.0e-06;
   (relax_data -> max_iter)         = 1000;
   (relax_data -> rel_change)       = 0;
   (relax_data -> zero_guess)       = 0;
   (relax_data -> weight)           = 1.0;
   (relax_data -> num_nodesets)     = 0;
   (relax_data -> nodeset_sizes)    = NULL;
   (relax_data -> nodeset_ranks)    = NULL;
   (relax_data -> nodeset_strides)  = NULL;
   (relax_data -> nodeset_indices)  = NULL;
   (relax_data -> diag_rank)        = NULL;
   (relax_data -> t)                = NULL;
   (relax_data -> A_loc)            = NULL;
   (relax_data -> x_loc)            = NULL;
   (relax_data -> Ap)               = NULL;
   (relax_data -> bp)               = NULL;
   (relax_data -> xp)               = NULL;
   (relax_data -> tp)               = NULL;
   (relax_data -> comm_handle)      = NULL;
   (relax_data -> svec_compute_pkgs)= NULL;
   (relax_data -> compute_pkgs)     = NULL;

   hypre_SetIndex(stride, 1, 1, 1);
   hypre_SetIndex(indices[0], 0, 0, 0);
   hypre_NodeRelaxSetNumNodesets((void *) relax_data, 1);
   hypre_NodeRelaxSetNodeset((void *) relax_data, 0, 1, stride, indices);

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxDestroy( void *relax_vdata )
{
   hypre_NodeRelaxData  *relax_data = relax_vdata;
   HYPRE_Int             i,vi;
   HYPRE_Int             ierr = 0;
   HYPRE_Int             nvars;

   if (relax_data)
   {
      nvars = hypre_SStructPMatrixNVars(relax_data -> A);
      for (i = 0; i < (relax_data -> num_nodesets); i++)
      {
         hypre_TFree(relax_data -> nodeset_indices[i]);
         for (vi = 0; vi < nvars; vi++)
         {
            hypre_ComputePkgDestroy(relax_data -> svec_compute_pkgs[i][vi]);
         }
         hypre_TFree(relax_data -> svec_compute_pkgs[i]);
         hypre_ComputePkgDestroy(relax_data -> compute_pkgs[i]);
      }
      hypre_TFree(relax_data -> nodeset_sizes);
      hypre_TFree(relax_data -> nodeset_ranks);
      hypre_TFree(relax_data -> nodeset_strides);
      hypre_TFree(relax_data -> nodeset_indices);
      hypre_SStructPMatrixDestroy(relax_data -> A);
      hypre_SStructPVectorDestroy(relax_data -> b);
      hypre_SStructPVectorDestroy(relax_data -> x);
      hypre_TFree(relax_data -> svec_compute_pkgs);
      hypre_TFree(relax_data -> comm_handle);
      hypre_TFree(relax_data -> compute_pkgs);
      hypre_SStructPVectorDestroy(relax_data -> t);

      hypre_TFree(relax_data -> x_loc);
      hypre_TFree((relax_data ->A_loc)[0]);
      hypre_TFree(relax_data -> A_loc);
      hypre_TFree(relax_data -> bp);
      hypre_TFree(relax_data -> xp);
      hypre_TFree(relax_data -> tp);
      for (vi = 0; vi < nvars; vi++)
      {
         hypre_TFree((relax_data -> Ap)[vi]);
         hypre_TFree((relax_data -> diag_rank)[vi]);
      }
      hypre_TFree(relax_data -> Ap);
      hypre_TFree(relax_data -> diag_rank);

      hypre_FinalizeTiming(relax_data -> time_index);
      hypre_TFree(relax_data);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetup(  void                 *relax_vdata,
                       hypre_SStructPMatrix *A,
                       hypre_SStructPVector *b,
                       hypre_SStructPVector *x           )
{
   hypre_NodeRelaxData   *relax_data = relax_vdata;

   HYPRE_Int              num_nodesets    = (relax_data -> num_nodesets);
   HYPRE_Int             *nodeset_sizes   = (relax_data -> nodeset_sizes);
   hypre_Index           *nodeset_strides = (relax_data -> nodeset_strides);
   hypre_Index          **nodeset_indices = (relax_data -> nodeset_indices);
   hypre_SStructPVector  *t;
   HYPRE_Int            **diag_rank;
   double               **A_loc;
   double                *x_loc;
   double              ***Ap;
   double               **bp;
   double               **xp;
   double               **tp;

   hypre_ComputeInfo     *compute_info;
   hypre_ComputePkg     **compute_pkgs;
   hypre_ComputePkg    ***svec_compute_pkgs;
   hypre_CommHandle     **comm_handle;

   hypre_Index            diag_index;
   hypre_IndexRef         stride;
   hypre_IndexRef         index;
                       
   hypre_StructGrid      *sgrid;

   hypre_StructStencil   *sstencil;
   hypre_Index           *sstencil_shape;
   HYPRE_Int              sstencil_size;

   hypre_StructStencil   *sstencil_union;
   hypre_Index           *sstencil_union_shape;
   HYPRE_Int              sstencil_union_count;

   hypre_BoxArrayArray   *orig_indt_boxes;
   hypre_BoxArrayArray   *orig_dept_boxes;
   hypre_BoxArrayArray   *box_aa;
   hypre_BoxArray        *box_a;
   hypre_Box             *box;
   HYPRE_Int              box_aa_size;
   HYPRE_Int              box_a_size;
   hypre_BoxArrayArray   *new_box_aa;
   hypre_BoxArray        *new_box_a;
   hypre_Box             *new_box;

   double                 scale;
   HYPRE_Int              frac;

   HYPRE_Int              i, j, k, p, m, s, compute_i;

   HYPRE_Int              vi, vj;
   HYPRE_Int              nvars;
   HYPRE_Int              dim;

   HYPRE_Int              ierr = 0;
                       
   /*----------------------------------------------------------
    * Set up the temp vector
    *----------------------------------------------------------*/

   if ((relax_data -> t) == NULL)
   {
      ierr = hypre_SStructPVectorCreate(hypre_SStructPVectorComm(b),
                                        hypre_SStructPVectorPGrid(b),
                                        &t);
      hypre_SStructPVectorInitialize(t);
      hypre_SStructPVectorAssemble(t);
      (relax_data -> t) = t;
   }

   /*----------------------------------------------------------
    * Find the matrix diagonals, use diag_rank[vi][vj] = -1 to
    * mark that the coresponding StructMatrix is NULL.
    *----------------------------------------------------------*/

   nvars = hypre_SStructPMatrixNVars(A);

   diag_rank = hypre_CTAlloc(HYPRE_Int *, nvars);
   for (vi = 0; vi < nvars; vi++)
      {
      diag_rank[vi] = hypre_CTAlloc(HYPRE_Int, nvars);
      for (vj = 0; vj < nvars; vj++)
      {
         if (hypre_SStructPMatrixSMatrix(A, vi, vj) != NULL)
         {
            sstencil = hypre_SStructPMatrixSStencil(A, vi, vj);
            hypre_SetIndex(diag_index, 0, 0, 0);
            diag_rank[vi][vj] = 
               hypre_StructStencilElementRank(sstencil, diag_index);
         }
         else
         {
            diag_rank[vi][vj] = -1; 
         }
      }
   }

   /*----------------------------------------------------------
    * Allocate storage used to invert local diagonal blocks
    *----------------------------------------------------------*/

   x_loc    = hypre_TAlloc(double   , hypre_NumThreads()*nvars);
   A_loc    = hypre_TAlloc(double  *, hypre_NumThreads()*nvars);
   A_loc[0] = hypre_TAlloc(double   , hypre_NumThreads()*nvars*nvars);
   for (vi = 1; vi < hypre_NumThreads()*nvars; vi++)
   {
      A_loc[vi] = A_loc[0] + vi*nvars;
   }

   /* Allocate pointers for vector and matrix */
   bp = hypre_TAlloc(double  *, nvars);
   xp = hypre_TAlloc(double  *, nvars);
   tp = hypre_TAlloc(double  *, nvars);
   Ap = hypre_TAlloc(double **, nvars);
   for (vi = 0; vi < nvars; vi++)
   {
      Ap[vi] = hypre_TAlloc(double  *, nvars);
   }

   /*----------------------------------------------------------
    * Set up the compute packages for each nodeset
    *----------------------------------------------------------*/

   sgrid = hypre_StructMatrixGrid(hypre_SStructPMatrixSMatrix(A, 0, 0));
   dim = hypre_StructStencilDim(
                          hypre_SStructPMatrixSStencil(A, 0, 0));

   compute_pkgs = hypre_CTAlloc(hypre_ComputePkg *, num_nodesets);
   svec_compute_pkgs = hypre_CTAlloc(hypre_ComputePkg **, num_nodesets);
   comm_handle = hypre_CTAlloc(hypre_CommHandle *, nvars);

   for (p = 0; p < num_nodesets; p++)
   {
      /*----------------------------------------------------------
       * Set up the compute packages to define sends and recieves
       * for each struct_vector (svec_compute_pkgs) and the compute
       * package to define independent and dependent computations
       * (compute_pkgs).
       *----------------------------------------------------------*/
      svec_compute_pkgs[p] = hypre_CTAlloc(hypre_ComputePkg *, nvars);

      for (vi = -1; vi < nvars; vi++)
      {

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
         sstencil_union_count = 0;
         if (vi == -1)
         {
            for (i = 0; i < nvars; i++)
            {
               for (vj = 0; vj < nvars; vj++)
               {
                  if (hypre_SStructPMatrixSMatrix(A,vj,i) != NULL)
                  {
                     sstencil = hypre_SStructPMatrixSStencil(A, vj, i);
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
               if (hypre_SStructPMatrixSMatrix(A,vj,vi) != NULL)
               {
                  sstencil = hypre_SStructPMatrixSStencil(A, vj, vi);
                  sstencil_union_count += hypre_StructStencilSize(sstencil);
               }
            }
         }
         sstencil_union_shape = hypre_CTAlloc(hypre_Index,
                                             sstencil_union_count);
         sstencil_union_count = 0;
         if (vi == -1)
         {
            for (i = 0; i < nvars; i++)
            {
               for (vj = 0; vj < nvars; vj++)
               {
                  if (hypre_SStructPMatrixSMatrix(A,vj,i) != NULL)
                  {
                     sstencil = hypre_SStructPMatrixSStencil(A, vj, i);
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
               if (hypre_SStructPMatrixSMatrix(A,vj,vi) != NULL)
               {
                  sstencil = hypre_SStructPMatrixSStencil(A, vj, vi);
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

         sstencil_union = hypre_StructStencilCreate(dim, 
                          sstencil_union_count, sstencil_union_shape);


         hypre_CreateComputeInfo(sgrid, sstencil_union, &compute_info);
         orig_indt_boxes = hypre_ComputeInfoIndtBoxes(compute_info);
         orig_dept_boxes = hypre_ComputeInfoDeptBoxes(compute_info);

         stride = nodeset_strides[p];

         for (compute_i = 0; compute_i < 2; compute_i++)
         {
            switch(compute_i)
            {
               case 0:
               box_aa = orig_indt_boxes;
               break;

               case 1:
               box_aa = orig_dept_boxes;
               break;
            }
            box_aa_size = hypre_BoxArrayArraySize(box_aa);
            new_box_aa = hypre_BoxArrayArrayCreate(box_aa_size);

            for (i = 0; i < box_aa_size; i++)
            {
               box_a = hypre_BoxArrayArrayBoxArray(box_aa, i);
               box_a_size = hypre_BoxArraySize(box_a);
               new_box_a = hypre_BoxArrayArrayBoxArray(new_box_aa, i);
               hypre_BoxArraySetSize(new_box_a,
                                     box_a_size * nodeset_sizes[p]);

               k = 0;
               for (m = 0; m < nodeset_sizes[p]; m++)
               {
                  index  = nodeset_indices[p][m];
   
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
   
            switch(compute_i)
            {
               case 0:
               hypre_ComputeInfoIndtBoxes(compute_info) = new_box_aa;
               break;
   
               case 1:
               hypre_ComputeInfoDeptBoxes(compute_info) = new_box_aa;
               break;
            }
         }

         hypre_CopyIndex(stride, hypre_ComputeInfoStride(compute_info));

         if (vi == -1)
         {
            hypre_ComputePkgCreate(compute_info,
                                   hypre_StructVectorDataSpace(
                                   hypre_SStructPVectorSVector(x, 0)),
                                   1, sgrid, &compute_pkgs[p]);
         }
         else
         {
            hypre_ComputePkgCreate(compute_info,
                                   hypre_StructVectorDataSpace(
                                   hypre_SStructPVectorSVector(x, vi)),
                                   1, sgrid, &svec_compute_pkgs[p][vi]);
         }
   
         hypre_BoxArrayArrayDestroy(orig_indt_boxes);
         hypre_BoxArrayArrayDestroy(orig_dept_boxes);

         hypre_StructStencilDestroy(sstencil_union);
      }
   }
   
   /*----------------------------------------------------------
    * Set up the relax data structure
    *----------------------------------------------------------*/

   hypre_SStructPMatrixRef(A, &(relax_data -> A));
   hypre_SStructPVectorRef(x, &(relax_data -> x));
   hypre_SStructPVectorRef(b, &(relax_data -> b));

   (relax_data -> diag_rank)    = diag_rank;
   (relax_data -> A_loc)    = A_loc;
   (relax_data -> x_loc)    = x_loc;
   (relax_data -> Ap)    = Ap;
   (relax_data -> bp)    = bp;
   (relax_data -> tp)    = tp;
   (relax_data -> xp)    = xp;
   (relax_data -> compute_pkgs) = compute_pkgs;
   (relax_data -> svec_compute_pkgs) = svec_compute_pkgs;
   (relax_data -> comm_handle) = comm_handle;

   /*-----------------------------------------------------
    * Compute flops 
    *-----------------------------------------------------*/

   scale = 0.0;
   for (p = 0; p < num_nodesets; p++)
   {
      stride = nodeset_strides[p];
      frac   = hypre_IndexX(stride);
      frac  *= hypre_IndexY(stride);
      frac  *= hypre_IndexZ(stride);
      scale += (nodeset_sizes[p] / frac);
   }
   /* REALLY Rough Estimate = num_nodes * nvar^3 */
   (relax_data -> flops) = scale * nvars * nvars * nvars *
                           hypre_StructVectorGlobalSize(
                           hypre_SStructPVectorSVector(x,0));

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelax(  void               *relax_vdata,
                  hypre_SStructPMatrix *A,
                  hypre_SStructPVector *b,
                  hypre_SStructPVector *x           )
{
   hypre_NodeRelaxData   *relax_data = relax_vdata;

   HYPRE_Int              max_iter         = (relax_data -> max_iter);
   HYPRE_Int              zero_guess       = (relax_data -> zero_guess);
   double                 weight           = (relax_data -> weight);
   HYPRE_Int              num_nodesets     = (relax_data -> num_nodesets);
   HYPRE_Int             *nodeset_ranks    = (relax_data -> nodeset_ranks);
   hypre_Index           *nodeset_strides  = (relax_data -> nodeset_strides);
   hypre_SStructPVector  *t                = (relax_data -> t);
   HYPRE_Int            **diag_rank        = (relax_data -> diag_rank);
   hypre_ComputePkg     **compute_pkgs     = (relax_data -> compute_pkgs);
   hypre_ComputePkg    ***svec_compute_pkgs= (relax_data ->
                                               svec_compute_pkgs);
   hypre_CommHandle     **comm_handle      = (relax_data -> comm_handle);

   hypre_ComputePkg      *compute_pkg;
   hypre_ComputePkg      *svec_compute_pkg;
                        
   hypre_BoxArrayArray   *compute_box_aa;
   hypre_BoxArray        *compute_box_a;
   hypre_Box             *compute_box;
                        
   hypre_Box             *A_data_box;
   hypre_Box             *b_data_box;
   hypre_Box             *x_data_box;
   hypre_Box             *t_data_box;
                        
   HYPRE_Int              Ai;
   HYPRE_Int              bi;
   HYPRE_Int              xi;
   HYPRE_Int              ti;
                        
   double               **tA_loc = (relax_data -> A_loc);
   double                *tx_loc = (relax_data -> x_loc);
   double               **A_loc;
   double                *x_loc;

   double              ***Ap = (relax_data -> Ap);
   double               **bp = (relax_data -> bp);
   double               **xp = (relax_data -> xp);
   double               **tp = (relax_data -> tp);

   hypre_StructMatrix    *A_block;
   hypre_StructVector    *x_block;

   hypre_IndexRef         stride;
   hypre_IndexRef         start;
   hypre_Index            loop_size;
                        
   hypre_StructStencil   *stencil;
   hypre_Index           *stencil_shape;
   HYPRE_Int              stencil_size;
                        
   HYPRE_Int              iter, p, compute_i, i, j, si;
   HYPRE_Int              loopi, loopj, loopk;
   HYPRE_Int              nodeset;

   HYPRE_Int              nvars;
   HYPRE_Int              vi, vj;

   HYPRE_Int              ierr = 0;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/

   hypre_BeginTiming(relax_data -> time_index);

   hypre_SStructPMatrixDestroy(relax_data -> A);
   hypre_SStructPVectorDestroy(relax_data -> b);
   hypre_SStructPVectorDestroy(relax_data -> x);
   hypre_SStructPMatrixRef(A, &(relax_data -> A));
   hypre_SStructPVectorRef(x, &(relax_data -> x));
   hypre_SStructPVectorRef(b, &(relax_data -> b));

   (relax_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         hypre_SStructPVectorSetConstantValues(x, 0.0);
      }

      hypre_EndTiming(relax_data -> time_index);
      return ierr;
   }

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   p    = 0;
   iter = 0;

   nvars = hypre_SStructPMatrixNVars(relax_data -> A);

   if (zero_guess)
   {
      if (num_nodesets > 1)
      {
         hypre_SStructPVectorSetConstantValues(x, 0.0);
      }
      nodeset = nodeset_ranks[p];
      compute_pkg = compute_pkgs[nodeset];
      stride = nodeset_strides[nodeset];

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         hypre_ForBoxArrayI(i, compute_box_aa)
            {
               compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

               A_data_box =
                  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(
                                     hypre_SStructPMatrixSMatrix(A,0,0)), i);
               b_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(
                                     hypre_SStructPVectorSVector(b,0)), i);
               x_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(
                                     hypre_SStructPVectorSVector(x,0)), i);

               for (vi = 0; vi < nvars; vi++)
               {
                  for (vj = 0; vj < nvars; vj++)
                  {
                     if (hypre_SStructPMatrixSMatrix(A,vi,vj) != NULL)
                     {
                        Ap[vi][vj] = hypre_StructMatrixBoxData(
                                      hypre_SStructPMatrixSMatrix(A,vi,vj),
                                      i, diag_rank[vi][vj]);
                     }
                  }
                  bp[vi] = hypre_StructVectorBoxData(
                            hypre_SStructPVectorSVector(b,vi), i);
                  xp[vi] = hypre_StructVectorBoxData(
                            hypre_SStructPVectorSVector(x,vi), i);
               }

               hypre_ForBoxI(j, compute_box_a)
                  {
                     compute_box = hypre_BoxArrayBox(compute_box_a, j);

                     start  = hypre_BoxIMin(compute_box);
                     hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                     hypre_BoxLoop3Begin(loop_size,
                                         A_data_box, start, stride, Ai,
                                         b_data_box, start, stride, bi,
                                         x_data_box, start, stride, xi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,bi,xi,vi,vj,x_loc,A_loc
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop3For(loopi, loopj, loopk, Ai, bi, xi)
                        {
                           A_loc = &tA_loc[hypre_BoxLoopBlock()*nvars];
                           x_loc = &tx_loc[hypre_BoxLoopBlock()*nvars];
                           /*------------------------------------------------
                            * Copy rhs and matrix for diagonal coupling
                            * (intra-nodal) into local storage.
                            *----------------------------------------------*/
                           for (vi = 0; vi < nvars; vi++)
                           {
                              x_loc[vi] = bp[vi][bi];
                              for (vj = 0; vj < nvars; vj++)
                              {
                                 if (hypre_SStructPMatrixSMatrix(A,vi,vj)
                                                        != NULL)
                                 {
                                    A_loc[vi][vj] = Ap[vi][vj][Ai];
                                 }
                                 else
                                 {
                                   A_loc[vi][vj] = 0.0;
                                 }
                              }
                           }

                           /*------------------------------------------------
                            * Invert intra-nodal coupling 
                            *----------------------------------------------*/
                           ierr = gselim(A_loc[0], x_loc, nvars);

                           /*------------------------------------------------
                            * Copy solution from local storage.
                            *----------------------------------------------*/
                           for (vi = 0; vi < nvars; vi++)
                           {
                              xp[vi][xi] = x_loc[vi];
                           }

                        }
                     hypre_BoxLoop3End(Ai, bi, xi);
                  }
            }
      }

      if (weight != 1.0)
      {
         hypre_SStructPScale(weight, x);
      }

      p    = (p + 1) % num_nodesets;
      iter = iter + (p == 0);
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/

   while (iter < max_iter)
   {
      nodeset = nodeset_ranks[p];
      compute_pkg = compute_pkgs[nodeset];
      stride = nodeset_strides[nodeset];

      hypre_SStructPCopy(x, t);

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               for (vi = 0; vi < nvars; vi++)
               {
                  x_block = hypre_SStructPVectorSVector(x,vi);
                  xp[vi] = hypre_StructVectorData(x_block);
                  svec_compute_pkg = svec_compute_pkgs[nodeset][vi];
                  hypre_InitializeIndtComputations(svec_compute_pkg,
                       xp[vi], &comm_handle[vi]);
               }
               compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               for (vi = 0; vi < nvars; vi++)
               {
                  hypre_FinalizeIndtComputations(comm_handle[vi]);
               }
               compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         hypre_ForBoxArrayI(i, compute_box_aa)
            {
               compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

               A_data_box =
                  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(
                                     hypre_SStructPMatrixSMatrix(A,0,0)), i);
               b_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(
                                     hypre_SStructPVectorSVector(b,0)), i);
               x_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(
                                     hypre_SStructPVectorSVector(x,0)), i);
               t_data_box =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(
                                     hypre_SStructPVectorSVector(t,0)), i);

               for (vi = 0; vi < nvars; vi++)
               {
                  bp[vi] = hypre_StructVectorBoxData(
                            hypre_SStructPVectorSVector(b,vi), i);
                  tp[vi] = hypre_StructVectorBoxData(
                            hypre_SStructPVectorSVector(t,vi), i);
               }

               hypre_ForBoxI(j, compute_box_a)
                  {
                     compute_box = hypre_BoxArrayBox(compute_box_a, j);

                     start  = hypre_BoxIMin(compute_box);
                     hypre_BoxGetStrideSize(compute_box, stride, loop_size);

                     hypre_BoxLoop2Begin(loop_size,
                                         b_data_box, start, stride, bi,
                                         t_data_box, start, stride, ti);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,bi,ti,vi
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop2For(loopi, loopj, loopk, bi, ti)
                        {
                           /* Copy rhs into temp vector */ 
                           for (vi = 0; vi < nvars; vi++)
                           {
                              tp[vi][ti] = bp[vi][bi];
                           }
                        }
                     hypre_BoxLoop2End(bi, ti);

                     for (vi = 0; vi < nvars; vi++)
                     {
                        for (vj = 0; vj < nvars; vj++)
                        {
                           if (hypre_SStructPMatrixSMatrix(A,vi,vj) != NULL)
                           {
                            A_block = hypre_SStructPMatrixSMatrix(A,vi,vj);
                            x_block = hypre_SStructPVectorSVector(x,vj);
                            stencil = hypre_StructMatrixStencil(A_block);
                            stencil_shape = hypre_StructStencilShape(stencil);
                            stencil_size  = hypre_StructStencilSize(stencil);
                            for (si = 0; si < stencil_size; si++)
                            {
                               if (si != diag_rank[vi][vj])
                               {
                                  Ap[vi][vj] = hypre_StructMatrixBoxData(
                                                 A_block,i,si);
                                  xp[vj] = hypre_StructVectorBoxData(x_block,i)
                                         + hypre_BoxOffsetDistance(
                                              x_data_box, stencil_shape[si]);

                                  hypre_BoxLoop3Begin(loop_size,
                                                A_data_box, start, stride, Ai,
                                                x_data_box, start, stride, xi,
                                                t_data_box, start, stride, ti);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,xi,ti
#include "hypre_box_smp_forloop.h"
                                  hypre_BoxLoop3For(loopi,loopj,loopk,Ai,xi,ti)
                                  {
                                     tp[vi][ti] -= Ap[vi][vj][Ai] * xp[vj][xi];
                                  }
                                  hypre_BoxLoop3End(Ai, xi, ti);
                               }
                            }
                           }
                        }
                     }

                     for (vi = 0; vi < nvars; vi++)
                     {
                        for (vj = 0; vj < nvars; vj++)
                        {
                           if (hypre_SStructPMatrixSMatrix(A,vi,vj) != NULL)
                           {
                              Ap[vi][vj] = hypre_StructMatrixBoxData(
                                        hypre_SStructPMatrixSMatrix(A,vi,vj),
                                        i, diag_rank[vi][vj]);
                           }
                        }
                     }

                     hypre_BoxLoop2Begin(loop_size,
                                         A_data_box, start, stride, Ai,
                                         t_data_box, start, stride, ti);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,Ai,ti,vi,vj,x_loc,A_loc
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop2For(loopi, loopj, loopk, Ai, ti)
                        {
                           A_loc = &tA_loc[hypre_BoxLoopBlock()*nvars];
                           x_loc = &tx_loc[hypre_BoxLoopBlock()*nvars];
                           /*------------------------------------------------
                            * Copy rhs and matrix for diagonal coupling
                            * (intra-nodal) into local storage.
                            *----------------------------------------------*/
                           for (vi = 0; vi < nvars; vi++)
                           {
                              x_loc[vi] = tp[vi][ti];
                              for (vj = 0; vj < nvars; vj++)
                              {
                                 if (hypre_SStructPMatrixSMatrix(A,vi,vj)
                                                        != NULL)
                                 {
                                    A_loc[vi][vj] = Ap[vi][vj][Ai];
                                 }
                                 else
                                 {
                                   A_loc[vi][vj] = 0.0;
                                 }
                              }
                           }

                           /*------------------------------------------------
                            * Invert intra-nodal coupling
                            *----------------------------------------------*/
                           ierr = gselim(A_loc[0], x_loc, nvars);

                           /*------------------------------------------------
                            * Copy solution from local storage.
                            *----------------------------------------------*/
                           for (vi = 0; vi < nvars; vi++)
                           {
                              tp[vi][ti] = x_loc[vi];
                           }

                        }
                     hypre_BoxLoop2End(Ai, ti);
                  }
            }
      }

      if (weight != 1.0)
      {
         hypre_SStructPScale((1.0 - weight), x);
         hypre_SStructPAxpy(weight, t, x);
      }
      else
      {
         hypre_SStructPCopy(t, x);
      }

      p    = (p + 1) % num_nodesets;
      iter = iter + (p == 0);
   }
   
   (relax_data -> num_iterations) = iter;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   hypre_IncFLOPCount(relax_data -> flops);
   hypre_EndTiming(relax_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetTol( void   *relax_vdata,
                       double  tol         )
{
   hypre_NodeRelaxData *relax_data = relax_vdata;
   HYPRE_Int            ierr = 0;

   (relax_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetMaxIter( void *relax_vdata,
                           HYPRE_Int   max_iter    )
{
   hypre_NodeRelaxData *relax_data = relax_vdata;
   HYPRE_Int            ierr = 0;

   (relax_data -> max_iter) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetZeroGuess( void *relax_vdata,
                             HYPRE_Int   zero_guess  )
{
   hypre_NodeRelaxData *relax_data = relax_vdata;
   HYPRE_Int            ierr = 0;

   (relax_data -> zero_guess) = zero_guess;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetWeight
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetWeight( void    *relax_vdata,
                          double   weight      )
{
   hypre_NodeRelaxData *relax_data = relax_vdata;
   HYPRE_Int            ierr = 0;

   (relax_data -> weight) = weight;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetNumNodesets
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetNumNodesets( void *relax_vdata,
                               HYPRE_Int   num_nodesets )
{
   hypre_NodeRelaxData *relax_data = relax_vdata;
   HYPRE_Int            i;
   HYPRE_Int            ierr = 0;

   /* free up old nodeset memory */
   for (i = 0; i < (relax_data -> num_nodesets); i++)
   {
      hypre_TFree(relax_data -> nodeset_indices[i]);
   }
   hypre_TFree(relax_data -> nodeset_sizes);
   hypre_TFree(relax_data -> nodeset_ranks);
   hypre_TFree(relax_data -> nodeset_strides);
   hypre_TFree(relax_data -> nodeset_indices);

   /* alloc new nodeset memory */
   (relax_data -> num_nodesets)    = num_nodesets;
   (relax_data -> nodeset_sizes)   = hypre_TAlloc(HYPRE_Int, num_nodesets);
   (relax_data -> nodeset_ranks)   = hypre_TAlloc(HYPRE_Int, num_nodesets);
   (relax_data -> nodeset_strides) = hypre_TAlloc(hypre_Index, num_nodesets);
   (relax_data -> nodeset_indices) = hypre_TAlloc(hypre_Index *,
                                                   num_nodesets);
   for (i = 0; i < num_nodesets; i++)
   {
      (relax_data -> nodeset_sizes[i]) = 0;
      (relax_data -> nodeset_ranks[i]) = i;
      (relax_data -> nodeset_indices[i]) = NULL;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetNodeset
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetNodeset( void        *relax_vdata,
                           HYPRE_Int    nodeset,
                           HYPRE_Int    nodeset_size,
                           hypre_Index  nodeset_stride,
                           hypre_Index *nodeset_indices )
{
   hypre_NodeRelaxData *relax_data = relax_vdata;
   HYPRE_Int            i;
   HYPRE_Int            ierr = 0;

   /* free up old nodeset memory */
   hypre_TFree(relax_data -> nodeset_indices[nodeset]);

   /* alloc new nodeset memory */
   (relax_data -> nodeset_indices[nodeset]) =
      hypre_TAlloc(hypre_Index, nodeset_size);

   (relax_data -> nodeset_sizes[nodeset]) = nodeset_size;
   hypre_CopyIndex(nodeset_stride,
                   (relax_data -> nodeset_strides[nodeset]));
   for (i = 0; i < nodeset_size; i++)
   {
      hypre_CopyIndex(nodeset_indices[i],
                      (relax_data -> nodeset_indices[nodeset][i]));
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetNodesetRank
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetNodesetRank( void *relax_vdata,
                               HYPRE_Int   nodeset,
                               HYPRE_Int   nodeset_rank )
{
   hypre_NodeRelaxData *relax_data = relax_vdata;
   HYPRE_Int            ierr = 0;

   (relax_data -> nodeset_ranks[nodeset]) = nodeset_rank;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_NodeRelaxSetTempVec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NodeRelaxSetTempVec( void                 *relax_vdata,
                           hypre_SStructPVector *t           )
{
   hypre_NodeRelaxData  *relax_data = relax_vdata;
   HYPRE_Int             ierr = 0;

   hypre_SStructPVectorDestroy(relax_data -> t);
   hypre_SStructPVectorRef(t, &(relax_data -> t));

   return ierr;
}

