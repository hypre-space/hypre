/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/




#include "_hypre_parcsr_ls.h"
#include "Common.h"

#define SV_DEBUG 0


/******************************************************************************
 hypre_BoomerAMGSmoothInterpVectors-

 *apply hybrid GS smoother to the interp vectors

*******************************************************************************/

HYPRE_Int hypre_BoomerAMGSmoothInterpVectors(hypre_ParCSRMatrix *A,
                                             HYPRE_Int num_smooth_vecs,
                                             hypre_ParVector **smooth_vecs,
                                             HYPRE_Int smooth_steps)

{
   HYPRE_Int i, j;

   hypre_ParVector *f, *v, *z;
   hypre_ParVector *new_vector;

   if (num_smooth_vecs == 0)
   {
      return hypre_error_flag;
   }

   if (smooth_steps)
   {
      v = hypre_ParVectorInRangeOf( A);
      f = hypre_ParVectorInRangeOf( A);
      z = hypre_ParVectorInRangeOf( A);

      hypre_ParVectorSetConstantValues(f, 0.0);

      for (i = 0; i < num_smooth_vecs; i++)
      {
         new_vector = smooth_vecs[i];

         for (j = 0; j < smooth_steps; j++)
         {
            hypre_BoomerAMGRelax(A, f, NULL, 3, 0, 1.0, 1.0, NULL, new_vector, v, z);
         }
      }

      hypre_ParVectorDestroy(v);
      hypre_ParVectorDestroy(f);
      hypre_ParVectorDestroy(z);

   }

   return hypre_error_flag;
}

/******************************************************************************

 hypre_BoomerAMGCoarsenInterpVectors:

 *this routine for "coarsening" the interp vectors

 *expand_level = 1, means that the new smooth vecs need to be expanded
 to fit the new num functions (this typically happends at
 interp_first_level)


 ******************************************************************************/

HYPRE_Int
hypre_BoomerAMGCoarsenInterpVectors( hypre_ParCSRMatrix *P,
                                     HYPRE_Int num_smooth_vecs,
                                     hypre_ParVector **smooth_vecs,
                                     HYPRE_Int *CF_marker,
                                     hypre_ParVector ***new_smooth_vecs,
                                     HYPRE_Int expand_level,
                                     HYPRE_Int num_functions)
{

   HYPRE_Int i, j, k;

   HYPRE_BigInt  n_new = hypre_ParCSRMatrixGlobalNumCols(P);

   HYPRE_BigInt *starts = hypre_ParCSRMatrixColStarts(P);

   HYPRE_Int    n_old_local;
   HYPRE_Int    counter;

   HYPRE_Int orig_nf;

   HYPRE_Real *old_vector_data;
   HYPRE_Real *new_vector_data;

   MPI_Comm   comm   = hypre_ParCSRMatrixComm(P);

   hypre_ParVector *old_vector;
   hypre_ParVector *new_vector;

   hypre_ParVector **new_vector_array;

   if (num_smooth_vecs == 0)
   {
      return hypre_error_flag;
   }

   new_vector_array = hypre_CTAlloc(hypre_ParVector*,  num_smooth_vecs, HYPRE_MEMORY_HOST);

   /* get the size of the vector we are coarsening */
   old_vector = smooth_vecs[0];
   n_old_local = hypre_VectorSize(hypre_ParVectorLocalVector(old_vector));

   for (i = 0; i < num_smooth_vecs; i++)
   {
      new_vector = hypre_ParVectorCreate(comm, n_new, starts);
      hypre_ParVectorInitialize(new_vector);
      new_vector_data = hypre_VectorData(hypre_ParVectorLocalVector(new_vector));

      old_vector = smooth_vecs[i];
      old_vector_data = hypre_VectorData(hypre_ParVectorLocalVector(old_vector));

      /* copy coarse data to new vector*/
      counter = 0;
      /* need to do differently for the expansion level because the old vector is
         to small (doesn't have new dofs) */
      if (expand_level)
      {
         orig_nf = num_functions - num_smooth_vecs;
         /*  nodal coarsening, so just check the first dof in each
             node, i.e. loop through nodes */
         for (j = 0; j < n_old_local; j += orig_nf)
         {
            if (CF_marker[j] >= 0)
            {
               for (k = 0; k < orig_nf; k++) /* orig dofs */
               {
                  new_vector_data[counter++] = old_vector_data[j + k];
               }
               for (k = 0; k < num_smooth_vecs; k++ ) /* new dofs */
               {
                  if (k == i)
                  {
                     new_vector_data[counter++] = 1.0;
                  }
                  else
                  {
                     new_vector_data[counter++] = 0.0;
                  }
                  /* there is nothing to copy, so just put a 1.0 or 0.0 here
                     - then the next level works
                     correctly - this value not used anyhow - but now it is nice
                     if printed for matlab */
               }
            }
         }
      }
      else /* normal level */
      {
         for (j = 0; j < n_old_local; j++)
         {
            if (CF_marker[j] >= 0)
            {
               new_vector_data[counter++] = old_vector_data[j];
            }
         }
      }

      /*assign new_vector to vector array */
      new_vector_array[i] = new_vector;
   }

   *new_smooth_vecs = new_vector_array;

   return hypre_error_flag;
}



/******************************************************************************

  hypre_BoomerAMG_GMExpandInterp-

 routine for updating the interp operator to interpolate the supplied
 smooth vectors by expanding P in a SA-ish manner This is the GM
 approach as described in Baker,Kolev and Yang "Improving AMG
 interpolation operators for linear elasticity problems"

 *MUST USE NODAL COARSENING! (and so unknowns interlaced)

 *NOTE: we assume that we are adding 1 dof for 2D and 3 dof for 3D

 P = [P Q]

  variant = 1: (GM approach 1) Q_ij = P_ij*v_i/sum_j(P_ij),
                where v is the smooth vec

  variant  = 2: GM approach 2).: Q_ij = P_ij(v_i/sum_j(P_ij) - vc_j)
                (vc is coarse version of v)
                this variant we must call on all levels
                here we modify P_s (P corresponding to new unknowns)

 *if level = first_level - add the new dofs ocrresponding to the number of
 interp vecs - otherwise, the unknowns are there and we are just
 augmenting the matrix

 *note: changes num_functions and updates dof_array if level = 0

 *abs_trunc - don't add elements to Q less than abs_truc (we don't use the
 regular interp truncation function because it rescales the rows, which we
 don't want to do that)


 ******************************************************************************/

HYPRE_Int
hypre_BoomerAMG_GMExpandInterp( hypre_ParCSRMatrix *A,
                                hypre_ParCSRMatrix **P,
                                HYPRE_Int num_smooth_vecs,
                                hypre_ParVector **smooth_vecs,
                                HYPRE_Int *nf,
                                HYPRE_Int *dof_func,
                                hypre_IntArray **coarse_dof_func,
                                HYPRE_Int variant,
                                HYPRE_Int level,
                                HYPRE_Real abs_trunc,
                                HYPRE_Real *weights,
                                HYPRE_Int q_max,
                                HYPRE_Int *CF_marker,
                                HYPRE_Int interp_vec_first_level)
{

   HYPRE_Int i, j, k;

   hypre_ParCSRMatrix *new_P;

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(*P);
   HYPRE_Real      *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int       *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int       *P_diag_j = hypre_CSRMatrixJ(P_diag);
   HYPRE_Int        num_rows_P = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int        num_cols_P = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int        P_diag_size = P_diag_i[num_rows_P];

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(*P);
   HYPRE_Int       *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int        P_offd_size = P_offd_i[num_rows_P];

   HYPRE_Real      *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int       *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Int        num_cols_P_offd = hypre_CSRMatrixNumCols(P_offd);

   HYPRE_BigInt    *col_map_offd_P = hypre_ParCSRMatrixColMapOffd(*P);

   HYPRE_BigInt    *col_starts = hypre_ParCSRMatrixColStarts(*P);

   HYPRE_BigInt    *new_col_map_offd_P = NULL;


   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(*P);

   MPI_Comm         comm;

   HYPRE_Int        num_sends;
   HYPRE_Int        new_nnz_diag, new_nnz_offd, orig_diag_start, orig_offd_start;
   HYPRE_Int        j_diag_pos, j_offd_pos;
   HYPRE_Int        nnz_diag, nnz_offd, fcn_num, num_elements;
   HYPRE_Int        num_diag_elements, num_offd_elements;

   HYPRE_Int       *P_diag_j_new, *P_diag_i_new, *P_offd_i_new, *P_offd_j_new;
   HYPRE_BigInt    *P_offd_j_big = NULL;
   HYPRE_Real      *P_diag_data_new, *P_offd_data_new;

   HYPRE_Int        nv, ncv, ncv_peru;
   HYPRE_Int        new_ncv;
   HYPRE_Int        new_nf = *nf;

   HYPRE_Int        myid = 0, num_procs = 1, p_count_diag, p_count_offd;

   hypre_ParVector *vector;

   HYPRE_Real      *vec_data;
   HYPRE_Real       row_sum;
   HYPRE_Real      *dbl_buf_data;
   HYPRE_Real      *smooth_vec_offd = NULL;
   HYPRE_Real      *offd_vec_data;

   HYPRE_Int        orig_nf;
   HYPRE_BigInt     new_col_starts[2];
   HYPRE_Int        num_functions = *nf;
   HYPRE_Int       *c_dof_func = hypre_IntArrayData(*coarse_dof_func);
   HYPRE_Int        modify = 0;
   HYPRE_Int        add_q = 0;

   HYPRE_Real       value;
   HYPRE_Real       trunc_value = 0.0;
   HYPRE_Real       theta_2D[] = {.5, .5};
   HYPRE_Real       theta_3D[] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

   HYPRE_Real      *theta;

   HYPRE_Int        q_count;
   HYPRE_Int        use_trunc_data = 0;

   HYPRE_Real      *q_data = NULL;
   HYPRE_Real      *q_trunc_data = NULL;

   HYPRE_Int       *is_q = NULL;
   HYPRE_Int        q_alloc = 0;
   HYPRE_BigInt    *aux_j = NULL;
   HYPRE_Real      *aux_data = NULL;
   HYPRE_Int       *is_diag = NULL;

   HYPRE_Int       *col_map;
   HYPRE_Int       *coarse_to_fine;
   HYPRE_Int        coarse_counter;
   HYPRE_Int        fine_index = 0;
   HYPRE_Int        index;
   HYPRE_BigInt     big_index, big_new_col, cur_col, g_nc;
   HYPRE_Int        new_col;

   HYPRE_Int *num_lost_sv = NULL;
   HYPRE_Int *q_count_sv = NULL;
   HYPRE_Int *lost_counter_q_sv = NULL;
   HYPRE_Real *lost_value_sv = NULL;
   HYPRE_Real *q_dist_value_sv = NULL;

   HYPRE_MemoryLocation memory_location_P = hypre_ParCSRMatrixMemoryLocation(A);

   /* only doing 2 variants */
   if (variant < 1 || variant > 2)
   {
      variant = 2;
   }


   /* variant 2 needs off proc sv data (Variant 1 needs it if we
    * use_truc_data = 1 )*/

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate ( *P );
      comm_pkg = hypre_ParCSRMatrixCommPkg(*P);

   }

   comm   = hypre_ParCSRCommPkgComm(comm_pkg);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

#if SV_DEBUG
   {
      char new_file[80];

      hypre_CSRMatrix *P_CSR = NULL;
      hypre_Vector *sv = NULL;

      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(*P);

      if (!myid)
      {
         hypre_sprintf(new_file, "%s.level.%d", "P_new_orig", level );
         if (P_CSR)
         {
            hypre_CSRMatrixPrint(P_CSR, new_file);
         }

      }

      hypre_CSRMatrixDestroy(P_CSR);

      if (level == interp_vec_first_level || variant == 2)
      {
         for (i = 0; i < num_smooth_vecs; i++)
         {
            sv = hypre_ParVectorToVectorAll(smooth_vecs[i]);

            if (!myid)
            {
               hypre_sprintf(new_file, "%s.%d.level.%d", "smoothvec", i, level );
               if (sv)
               {
                  hypre_SeqVectorPrint(sv, new_file);
               }
            }

            hypre_SeqVectorDestroy(sv);

         }
      }

      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
      if (!myid)
      {
         hypre_sprintf(new_file, "%s.level.%d", "A", level );
         if (P_CSR)
         {
            hypre_CSRMatrixPrint(P_CSR, new_file);
         }
      }

      hypre_CSRMatrixDestroy(P_CSR);

   }

#endif

   /*initialize */
   nv = num_rows_P;
   ncv = num_cols_P;
   nnz_diag = P_diag_size;
   nnz_offd = P_offd_size;


   /* add Q? */
   /* only on first level for variants other than 2 */
   if (variant == 2 || level == interp_vec_first_level)
   {
      add_q = 1;
   }


   /* modify P_s? */
   if (variant == 2)
   {
      modify = 1;
   }

   /* use different values to truncate? */
   if (variant == 1 )
   {
      use_trunc_data = 1;
   }

   /* Note: we assume a NODAL coarsening */

   /* First we need to make room for the new entries to P*/

   /*number of coarse variables for each unknown */
   ncv_peru = ncv / num_functions;

   if (level == interp_vec_first_level)
   {
      orig_nf = num_functions;
      /*orig_ncv = ncv;*/
   }
   else /* on deeper levels, need to know orig sizes (without new
         * dofs) */
   {
      orig_nf = num_functions - num_smooth_vecs;
      /*orig_ncv = ncv - ncv_peru*num_smooth_vecs;*/
   }

   /*weights for P_s */
   if (modify)
   {
      if (weights == NULL)
      {
         if (orig_nf == 2)
         {
            theta = theta_2D;
         }
         else
         {
            theta = theta_3D;
         }
      }
      else
      {
         theta = weights;
      }
   }


   /* if level = first_level, we need to fix the col numbering to leave
   * space for the new unknowns */

   col_map = hypre_CTAlloc(HYPRE_Int,  ncv, HYPRE_MEMORY_HOST);

   if (num_smooth_vecs && (level == interp_vec_first_level))
   {
      for (i = 0; i < ncv; i++)
      {
         /* map from old col number to new col number (leave spaces
          * for new unknowns to be interleaved */
         col_map[i] = i + (i / num_functions) * num_smooth_vecs;
      }
   }
   else
   {
      for (i = 0; i < ncv; i++)
      {
         /* map from old col number to new col number */
         col_map[i] = i;
      }
   }


   /* new number of nonzeros  - these are overestimates if level > first_level*/

   /* we will have the same sparsity in Q as in P */
   new_nnz_diag = nnz_diag + nnz_diag * num_smooth_vecs;
   new_nnz_offd = nnz_offd + nnz_offd * num_smooth_vecs;

   /* new number of coarse variables */
   if (level == interp_vec_first_level )
   {
      new_ncv = ncv + ncv_peru * num_smooth_vecs;
   }
   else
   {
      new_ncv = ncv;   /* unchanged on level > 0 */
   }

   P_diag_j_new    = hypre_CTAlloc(HYPRE_Int,  new_nnz_diag, memory_location_P);
   P_diag_data_new = hypre_CTAlloc(HYPRE_Real, new_nnz_diag, memory_location_P);
   P_diag_i_new    = hypre_CTAlloc(HYPRE_Int,  nv + 1,       memory_location_P);

   P_offd_j_big    = hypre_CTAlloc(HYPRE_BigInt, new_nnz_offd, HYPRE_MEMORY_HOST);
   P_offd_j_new    = hypre_CTAlloc(HYPRE_Int,    new_nnz_offd, memory_location_P);
   P_offd_data_new = hypre_CTAlloc(HYPRE_Real,   new_nnz_offd, memory_location_P);
   P_offd_i_new    = hypre_CTAlloc(HYPRE_Int,    nv + 1,       memory_location_P);

   P_diag_i_new[0] = P_diag_i[0];
   P_offd_i_new[0] = P_offd_i[0];

   /* if doing truncation of q, need to allocate q_data */
   if (add_q)
   {
      if (q_max > 0 || abs_trunc > 0.0)
      {
         /* what is max elements per row? */
         q_count = 0;
         for (i = 0; i < num_rows_P; i++)
         {
            num_elements = P_diag_i[i + 1] - P_diag_i[i];
            num_elements += (P_offd_i[i + 1] - P_offd_i[i]);

            if (num_elements > q_count) { q_count = num_elements; }
         }

         q_alloc =  q_count * (num_smooth_vecs + 1);
         q_data = hypre_CTAlloc(HYPRE_Real,  q_alloc, HYPRE_MEMORY_HOST);
         q_trunc_data = hypre_CTAlloc(HYPRE_Real,  q_alloc, HYPRE_MEMORY_HOST);
         is_q = hypre_CTAlloc(HYPRE_Int,  q_alloc, HYPRE_MEMORY_HOST);
         aux_data = hypre_CTAlloc(HYPRE_Real,  q_alloc, HYPRE_MEMORY_HOST);
         aux_j = hypre_CTAlloc(HYPRE_BigInt,  q_alloc, HYPRE_MEMORY_HOST);
         is_diag = hypre_CTAlloc(HYPRE_Int,  q_alloc, HYPRE_MEMORY_HOST);


         /* for truncation routines */
         q_count_sv = hypre_CTAlloc(HYPRE_Int,  num_smooth_vecs,
                                    HYPRE_MEMORY_HOST); /* number of new q entries for each smoothvec */
         num_lost_sv = hypre_CTAlloc(HYPRE_Int,  num_smooth_vecs, HYPRE_MEMORY_HOST); /* value dropped */
         lost_counter_q_sv = hypre_CTAlloc(HYPRE_Int,  num_smooth_vecs, HYPRE_MEMORY_HOST);
         lost_value_sv = hypre_CTAlloc(HYPRE_Real,  num_smooth_vecs,
                                       HYPRE_MEMORY_HOST); /* how many to drop */
         q_dist_value_sv = hypre_CTAlloc(HYPRE_Real,  num_smooth_vecs, HYPRE_MEMORY_HOST); ;
      }
   }

   /* create the coarse to fine*/
   coarse_to_fine = hypre_CTAlloc(HYPRE_Int,  ncv, HYPRE_MEMORY_HOST);
   coarse_counter = 0;
   for (i = 0; i < num_rows_P; i++)
   {
      if (CF_marker[i] >= 0)
      {
         coarse_to_fine[coarse_counter] = i;
         coarse_counter++;
      }
   }
   /* Get smooth vec components for the off-processor columns of P -
    * in smoothvec_offd*/
   if (num_procs > 1)
   {

      HYPRE_Int start, c_index;
      hypre_ParCSRCommHandle  *comm_handle;

      smooth_vec_offd =  hypre_CTAlloc(HYPRE_Real,  num_cols_P_offd * num_smooth_vecs, HYPRE_MEMORY_HOST);

      /* for now, do a seperate comm for each smooth vector */
      for (k = 0; k < num_smooth_vecs; k++)
      {

         vector = smooth_vecs[k];
         vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         dbl_buf_data = hypre_CTAlloc(HYPRE_Real,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                   num_sends), HYPRE_MEMORY_HOST);
         /* point into smooth_vec_offd */
         offd_vec_data =  smooth_vec_offd + k * num_cols_P_offd;

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               /* we need to do the coarse/fine conversion here */
               c_index = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
               fine_index = coarse_to_fine[c_index];

               dbl_buf_data[index++] = vec_data[fine_index];
            }
         }

         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, dbl_buf_data,
                                                     offd_vec_data);
         hypre_ParCSRCommHandleDestroy(comm_handle);

         hypre_TFree(dbl_buf_data, HYPRE_MEMORY_HOST);
      }
   }/*end num procs > 1 */


   /******** loop through rows - add P only to the rows of original
             functions. rows corresponding to new functions are either
             left as is or modified with weighted average of
             interpolation of original variables******/
   j_diag_pos = 0;
   j_offd_pos = 0;
   orig_diag_start = 0;
   orig_offd_start = 0;

   for (i = 0; i < num_rows_P; i++)
   {

      q_count = 0; /* number of entries of q added for this row */

      /* zero entries */
      for (j = 0; j < q_alloc; j++)
      {
         is_q[j] = 0;
         q_data[j] = 0.0;
         q_trunc_data[j] = 0.0;
      }

      /* get function num for this row */
      fcn_num = (HYPRE_Int) fmod(i, num_functions);

      if (fcn_num != dof_func[i])
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "WARNING - ROWS incorrectly ordered in hypre_BoomerAMG_GMExpandInterp!\n");
      }

      /* number of elements in row */
      num_diag_elements = P_diag_i[i + 1] - orig_diag_start;
      num_offd_elements = P_offd_i[i + 1] - orig_offd_start;

      /* loop through elements - copy each to new_P and create Q corresp to
         each smooth vec for the orig functions */
      p_count_diag = 0;
      p_count_offd = 0;

      /* original function dofs? */
      if (fcn_num < orig_nf)
      {

         row_sum = 1.0;
         if ((variant == 1 || variant == 2) && add_q)
         {
            /* calc. row sum */
            row_sum = 0.0;
            for (j = 0; j < num_diag_elements; j++)
            {
               row_sum +=  P_diag_data[orig_diag_start + j];
            }
            for (j = 0; j < num_offd_elements; j++)
            {
               row_sum +=  P_offd_data[orig_offd_start + j];
            }

            num_elements = num_diag_elements + num_offd_elements;

            if (num_elements && hypre_abs(row_sum) < 1e-15)
            {
               row_sum = 1.0;
            }
         }

         /**** first do diag elements *****/
         for (j = 0; j < num_diag_elements; j++)
         {

            /* first copy original entry corresponding to P */
            new_col = col_map[P_diag_j[orig_diag_start + j]];

            P_diag_j_new[j_diag_pos] = new_col;
            P_diag_data_new[j_diag_pos] = P_diag_data[orig_diag_start + j];
            j_diag_pos++;
            p_count_diag++;

            /* add Q ? (only add Q to original dofs )*/
            if (add_q)
            {
               /* the current column number */
               cur_col =  new_col;

               /* loop through the smooth vectors */
               for (k = 0; k < num_smooth_vecs; k++)
               {
                  /* point to the smooth vector */
                  vector = smooth_vecs[k];
                  vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

                  /* add an entry */

                  /* create a single new entry for Q*/
                  new_col = cur_col + (HYPRE_BigInt)((orig_nf - fcn_num) + k);

                  /* Determine the Q entry value*/
                  if (variant == 2)
                  {
                     /*HYPRE_Real dt;*/
                     /* Q: P_ij(v_i/row_sum - vc_j) - ** notice we use fine and coarse smooth vecs */
                     index = P_diag_j[orig_diag_start + j]; /* don't want to use col_map here
                                                             because we will index into
                                                             the smooth vector */
                     fine_index = coarse_to_fine[index];

                     /*dt =  P_diag_data[orig_diag_start+j];
                     dt = (vec_data[i]/row_sum - vec_data[fine_index]);*/
                     value = P_diag_data[orig_diag_start + j] *
                             (vec_data[i] / row_sum - vec_data[fine_index]);
                  }

                  else /* variant 1 */
                  {
                     /* create new entry for Q: P_ij*v_i /sum(P_ij)*/
                     value = (P_diag_data[orig_diag_start + j] * vec_data[i]) / row_sum;

                     if (abs_trunc > 0.0  && use_trunc_data )
                     {
                        fine_index = P_diag_j[orig_diag_start + j];
                        fine_index = coarse_to_fine[fine_index];

                        /* this is Tzanio's suggestion */
                        if (vec_data[fine_index] != 0.0 )
                        {
                           trunc_value = P_diag_data[orig_diag_start + j] * (vec_data[i]) /
                                         (vec_data[fine_index]);
                        }
                        else
                        {
                           trunc_value = P_diag_data[orig_diag_start + j] * (vec_data[i]);
                        }
                     }

                  } /* end of var 2 */

                  /* add the new entry to to P */
                  if (hypre_abs(value) > 0.0)
                  {
                     if (q_max > 0 || abs_trunc > 0.0)
                     {
                        if (use_trunc_data)
                        {
                           q_trunc_data[p_count_diag] = trunc_value;
                        } /* note that this goes in the p_count entry to line
                                                                        up with is_q */
                        is_q[p_count_diag] = k + 1; /* so we know which k*/
                        q_data[q_count++] = value;
                     }
                     P_diag_j_new[j_diag_pos] = new_col;
                     p_count_diag++;
                     P_diag_data_new[j_diag_pos++] = value;
                  }
               } /* end loop through smooth vecs */
            } /* end if add q */

         } /* end of loop through diag elements */

         /**** now do offd elements *****/
         p_count_offd = p_count_diag;
         for (j = 0; j < num_offd_elements; j++)
         {
            /* first copy original entry corresponding to P (but j
               needs to go back to regular numbering - will be
               compressed later when col_map_offd is generated*/
            index = P_offd_j[orig_offd_start + j];

            /* convert to the global col number using col_map_offd */
            big_index = col_map_offd_P[index];

            /*now adjust for the new dofs - since we are offd, can't
             * use col_map[index]*/
            if (num_smooth_vecs && (level == interp_vec_first_level))
            {
               big_new_col = big_index + (big_index / (HYPRE_BigInt)num_functions) * (HYPRE_BigInt)num_smooth_vecs;
            }
            else /* no adjustment */
            {
               big_new_col = big_index;
            }

            P_offd_j_big[j_offd_pos] = big_new_col;
            P_offd_data_new[j_offd_pos] = P_offd_data[orig_offd_start + j];
            j_offd_pos++;
            p_count_offd++;

            /* add Q ? (only add Q to original dofs )*/
            if (add_q)
            {
               /* the current column number */
               cur_col =  big_new_col;

               /* loop through the smooth vectors */
               for (k = 0; k < num_smooth_vecs; k++)
               {

                  /* point to the smooth vector */
                  vector = smooth_vecs[k];
                  vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));

                  /* point to the offd smooth vector */
                  offd_vec_data = smooth_vec_offd + k * num_cols_P_offd;

                  /* add an entry */

                  /* create a single new entry for Q*/
                  big_new_col = cur_col + (HYPRE_BigInt)((orig_nf - fcn_num) + k);

                  /* Determine the Q entry value*/
                  if (variant == 2 )
                  {
                     /*HYPRE_Real dt;*/
                     /* Q: P_ij(v_i/row_sum - vc_j) - * notice we use fine and coarse smooth vecs */

                     index = P_offd_j[orig_offd_start + j]; /* don't want to use col_map here
                                                             because we will index into
                                                             the smooth vector */

                     /* did thecoasrse/fine conversion when gathering from procs above */

                     /*dt =  P_offd_data[orig_offd_start+j];
                     dt = (vec_data[i]/row_sum - offd_vec_data[index]);*/

                     value  = P_offd_data[orig_offd_start + j] * (vec_data[i] / row_sum - offd_vec_data[index]);


                     /* dt = (vec_data[i]/row_sum - c_vec_data[cur_col]);
                        value  = P_offd_data[orig_offd_start+j]*(vec_data[i]/row_sum - c_vec_data[cur_col]);*/

                  }

                  else /* variant 1 */
                  {
                     /* create new entry for Q: P_ij*v_i /sum(P_ij)*/
                     value = (P_offd_data[orig_offd_start + j] * vec_data[i]) / row_sum;

                     if (abs_trunc > 0.0  && use_trunc_data )
                     {
                        index = P_offd_j[orig_offd_start + j];

                        /* this is Tzanio's suggestion */
                        if (offd_vec_data[fine_index] != 0.0 )
                        {
                           trunc_value = P_offd_data[orig_offd_start + j] * (vec_data[i]) /
                                         (offd_vec_data[index]);
                        }
                        else
                        {
                           trunc_value =  P_offd_data[orig_offd_start + j] * (vec_data[i]);
                        }
                     }

                  } /* end of var 2 */

                  /* add the new entry to to P */
                  if (hypre_abs(value) > 0.0)
                  {
                     if (q_max > 0 || abs_trunc > 0.0)
                     {
                        if (use_trunc_data)
                        {
                           q_trunc_data[p_count_offd] = trunc_value;
                        } /* note that this goes in the p_count entry to line
                                                                        up with is_q */
                        is_q[p_count_offd] = k + 1; /* so we know which k*/
                        q_data[q_count++] = value;
                     }
                     P_offd_j_big[j_offd_pos] = big_new_col;
                     p_count_offd++;
                     P_offd_data_new[j_offd_pos++] = value;
                  }
               } /* end loop through smooth vecs */
            } /* end if add q */

         } /* end of loop through offd elements */


      } /* end if original function dofs */
      else /* the new dofs */
      {

         if (modify) /* instead of copying, let's modify the P corresponding to the new dof -
                      * for 2D make it (P_u + P_v)/2....*/
         {
            HYPRE_Int m, m_pos;
            HYPRE_Real m_val;
            /*HYPRE_Real tmp;*/

            /**** first do diag elements *****/
            for (j = 0; j < num_diag_elements; j++)
            {
               m_val = 0.0;
               for (m = 0; m < orig_nf; m++)
               {
                  m_pos = P_diag_i[i - (fcn_num - m)] + j; /* recall - nodal coarsening */
                  /*tmp = P_diag_data[m_pos];*/
                  m_val += theta[m] * P_diag_data[m_pos];
               }

               /*m_val = m_val/orig_nf;*/
               P_diag_j_new[j_diag_pos] = P_diag_j[orig_diag_start + j];
               P_diag_data_new[j_diag_pos++] = m_val;
               p_count_diag++;
            }
            /**** now offd elements *****/
            /* recall that j needs to go back to regular numbering -
               will be compressed later when col_map_offd is
               generated*/
            p_count_offd = p_count_diag;
            for (j = 0; j < num_offd_elements; j++)
            {
               m_val = 0.0;
               for (m = 0; m < orig_nf; m++)
               {
                  m_pos = P_offd_i[i - (fcn_num - m)] + j; /* recall - nodal coarsening */
                  /*tmp = P_offd_data[m_pos];*/
                  m_val += theta[m] * P_offd_data[m_pos];
               }

               /*m_val = m_val/orig_nf;*/
               index = P_offd_j[orig_offd_start + j];
               big_index = col_map_offd_P[index];

               P_offd_j_big[j_offd_pos] = big_index;
               P_offd_data_new[j_offd_pos++] = m_val;
               p_count_offd++;
            }
         }
         else /* just copy original entry corresponding to P (so original result from
                 unk-based interp on new dof */
         {
            /**** first do diag elements *****/
            for (j = 0; j < num_diag_elements; j++)
            {
               P_diag_j_new[j_diag_pos] = P_diag_j[orig_diag_start + j];
               P_diag_data_new[j_diag_pos++] = P_diag_data[orig_diag_start + j];
               p_count_diag++;
            }
            /**** now offd elements *****/
            /* recall that j needs to go back to regular numbering -
               will be compressed later when col_map_offd is
               generated*/
            p_count_offd = p_count_diag;
            for (j = 0; j < num_offd_elements; j++)
            {
               index = P_offd_j[orig_offd_start + j];
               big_index = col_map_offd_P[index];

               P_offd_j_big[j_offd_pos] = big_index;
               P_offd_data_new[j_offd_pos++] = P_offd_data[orig_offd_start + j];
               p_count_offd++;
            }


         }
      }/* end of new dof stuff */


      /* adjust p_count_offd to not include diag*/
      p_count_offd = p_count_offd - p_count_diag;


      /* ANY TRUCATION ?*/

      if (add_q && q_count > 0 && (q_max > 0 || abs_trunc > 0.0))
      {

         HYPRE_Int tot_num_lost;
         HYPRE_Int new_diag_pos, new_offd_pos;
         HYPRE_Int j_counter, new_j_counter;
         HYPRE_Int cnt_new_q_data;
         HYPRE_Int lost_counter_diag, lost_counter_offd;
         HYPRE_Int which_q;

         /* initialize to zero*/
         for (j = 0; j < num_smooth_vecs; j++)
         {
            q_count_sv[j] = 0;
            num_lost_sv[j] = 0;
            lost_counter_q_sv[j] = 0;
            lost_value_sv[j] = 0.0;
            q_dist_value_sv[j] = 0.0;

         }

         /* absolute truncation ? */
         if (abs_trunc > 0.0)
         {
            cnt_new_q_data = 0;

            j_counter = 0;

            /* diag loop */
            for (j =  P_diag_i_new[i]; j <  P_diag_i_new[i] + p_count_diag; j++)
            {
               if (is_q[j_counter]) /* if > 0 then belongs to q */
               {
                  which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                  q_count_sv[which_q]++;

                  if (!use_trunc_data)
                  {
                     value = hypre_abs(P_diag_data_new[j]);
                  }
                  else
                  {
                     value = hypre_abs(q_trunc_data[j_counter]);
                  }

                  if (value < abs_trunc )
                  {
                     num_lost_sv[which_q]++;
                     lost_value_sv[which_q] += P_diag_data_new[j];
                  }
               }
               j_counter++;
            }
            /* offd loop  - don't reset j_counter*/
            for (j =  P_offd_i_new[i]; j <  P_offd_i_new[i] + p_count_offd; j++)
            {
               if (is_q[j_counter]) /* if > 0 then belongs to q */
               {
                  which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                  q_count_sv[which_q]++;

                  if (!use_trunc_data)
                  {
                     value = hypre_abs(P_offd_data_new[j]);
                  }
                  else
                  {
                     value = hypre_abs(q_trunc_data[j_counter]);
                  }

                  if (value < abs_trunc )
                  {
                     num_lost_sv[which_q] ++;
                     lost_value_sv[which_q] += P_offd_data_new[j];
                  }
               }
               j_counter++;
            }

            tot_num_lost = 0;
            for (j = 0; j < num_smooth_vecs; j++)
            {
               q_dist_value_sv[j] = 0.0;
               tot_num_lost +=  num_lost_sv[j];
            }


            /* now drop values and adjust remaining ones to keep rowsum const. */
            lost_counter_diag = 0;
            lost_counter_offd = 0;

            if (tot_num_lost)
            {
               /* figure out distribution value */
               for (j = 0; j < num_smooth_vecs; j++)
               {
                  if ((q_count_sv[j] - num_lost_sv[j]) > 0)
                  {
                     q_dist_value_sv[j] = lost_value_sv[j] / (q_count_sv[j] - num_lost_sv[j]);
                  }
               }

               j_counter = 0;
               new_j_counter = 0;

               /* diag entries  */
               new_diag_pos =  P_diag_i_new[i];
               for (j =  P_diag_i_new[i]; j <  P_diag_i_new[i] + p_count_diag; j++)
               {
                  if (!use_trunc_data)
                  {
                     value = hypre_abs(P_diag_data_new[j]);
                  }
                  else
                  {
                     value = hypre_abs(q_trunc_data[j_counter]);
                  }

                  if ( is_q[j_counter] && (value < abs_trunc) )
                  {
                     /* drop */
                     which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                     lost_counter_diag++;
                  }
                  else
                  {
                     /* keep  - and if it is a q value then add the distribution */
                     value =  P_diag_data_new[j];
                     if (is_q[j_counter])
                     {
                        which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                        value += q_dist_value_sv[which_q];
                        q_data[cnt_new_q_data++] = value;
                     }

                     P_diag_data_new[new_diag_pos] = value;
                     P_diag_j_new[new_diag_pos] = P_diag_j_new[j];
                     new_diag_pos++;
                     is_q[new_j_counter] = is_q[j_counter];
                     new_j_counter++;

                  }
                  j_counter++;
               }

               /* offd entries */
               new_offd_pos =  P_offd_i_new[i];
               for (j =  P_offd_i_new[i]; j <  P_offd_i_new[i] + p_count_offd; j++)
               {
                  if (!use_trunc_data)
                  {
                     value = hypre_abs(P_offd_data_new[j]);
                  }
                  else
                  {
                     value = hypre_abs(q_trunc_data[j_counter]);
                  }


                  if ( is_q[j_counter] && (value < abs_trunc) )
                  {
                     /* drop */
                     which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                     lost_counter_offd++;
                  }
                  else
                  {
                     /* keep  - and if it is a q value then add the distribution */
                     value =  P_offd_data_new[j];
                     if (is_q[j_counter])
                     {
                        which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                        value += q_dist_value_sv[which_q];
                        q_data[cnt_new_q_data++] = value;
                     }

                     P_offd_data_new[new_offd_pos] = value;
                     P_offd_j_big[new_offd_pos] = P_offd_j_big[j];
                     new_offd_pos++;
                     is_q[new_j_counter] = is_q[j_counter];
                     new_j_counter++;

                  }
                  j_counter++;
               }

               /* adjust p_count and j_pos */
               p_count_diag -= lost_counter_diag;
               p_count_offd -= lost_counter_offd;

               j_diag_pos -= lost_counter_diag;
               j_offd_pos -= lost_counter_offd;


               if (tot_num_lost != (lost_counter_diag + lost_counter_offd))
               {
                  hypre_error_w_msg(HYPRE_ERROR_GENERIC, "hypre_BoomerAMG_GMExpandInterp: 1st Truncation error \n");
               }

            }/* end of num_lost */

         }/* abs_trunc > 0 */

         /* max number of element truncation */
         if (q_max > 0)
         {

            HYPRE_Int p_count_tot;

            for (j = 0; j < num_smooth_vecs; j++)
            {
               q_count_sv[j] = 0;
               num_lost_sv[j] = 0;
               lost_value_sv[j] = 0.0;
            }

            /* copy all elements for the row into aux vectors and
             * count the q's for each smoothvec*/
            j_counter = 0;
            for (j = P_diag_i_new[i]; j < P_diag_i_new[i] + p_count_diag; j++)
            {
               if (is_q[j_counter]) /* if > 0 then belongs to q */
               {
                  which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                  q_count_sv[which_q]++;
               }

               aux_j[j_counter] = (HYPRE_BigInt)P_diag_j_new[j];
               aux_data[j_counter] = P_diag_data_new[j];
               is_diag[j_counter] = 1;

               j_counter++;
            }
            /* offd loop  - don't reset j_counter*/
            for (j =  P_offd_i_new[i]; j <  P_offd_i_new[i] + p_count_offd; j++)
            {
               if (is_q[j_counter]) /* if > 0 then belongs to q */
               {
                  which_q = is_q[j_counter] - 1; /* adjust to index into sv arrays */
                  q_count_sv[which_q]++;
               }
               aux_j[j_counter] = P_offd_j_big[j];
               aux_data[j_counter] = P_offd_data_new[j];
               is_diag[j_counter] = 0;

               j_counter++;
            }

            /* intitialize */
            tot_num_lost = 0;
            for (j = 0; j < num_smooth_vecs; j++)
            {
               /* new_num_q_sv[j] = q_count_sv[j]; */
               q_dist_value_sv[j] = 0.0;
               lost_value_sv[j] = 0.0;
               lost_counter_q_sv[j] = 0;
               num_lost_sv[j] =  q_count_sv[j] - q_max;;
               /* don't want num_lost to be negative */
               if (num_lost_sv[j] < 0)
               {
                  num_lost_sv[j] = 0;
               }
               tot_num_lost +=  num_lost_sv[j];
            }

            if (tot_num_lost > 0)
            {

               p_count_tot = p_count_diag + p_count_offd;

               /* only keep q_max elements - get rid of smallest */
               hypre_BigQsort4_abs(aux_data, aux_j, is_q, is_diag, 0, p_count_tot - 1);

               lost_counter_diag = 0;
               lost_counter_offd = 0;

               new_diag_pos =  P_diag_i_new[i];
               new_offd_pos =  P_offd_i_new[i];

               new_j_counter = 0;

               /* have to do diag and offd together because of sorting*/
               for (j =  0; j < p_count_tot; j++)
               {

                  which_q = 0;
                  if ( is_q[j] )
                  {
                     which_q = is_q[j] - 1; /* adjust to index into sv arrays */
                  }

                  if ( is_q[j] && (lost_counter_q_sv[which_q] < num_lost_sv[which_q]))
                  {
                     /*drop*/
                     lost_value_sv[which_q] += aux_data[j];

                     /* new_num_q_sv[which_q]--; */
                     lost_counter_q_sv[which_q]++;

                     /* check whether this is diag or offd element */
                     if (is_diag[j])
                     {
                        lost_counter_diag++;
                     }
                     else
                     {
                        lost_counter_offd++;
                     }

                     /* technically only need to do this the last time */
                     q_dist_value_sv[which_q] = lost_value_sv[which_q] / q_max;
                  }
                  else
                  {
                     /* keep and add the dist if necessart*/
                     value =  aux_data[j];
                     if (is_q[j])
                     {
                        which_q = is_q[j] - 1; /* adjust to index into sv arrays */
                        value += q_dist_value_sv[which_q];;
                     }
                     if (is_diag[j])
                     {
                        P_diag_data_new[new_diag_pos] = value;
                        P_diag_j_new[new_diag_pos] = (HYPRE_Int)aux_j[j];
                        new_diag_pos++;
                        is_q[new_j_counter] = is_q[j];
                        new_j_counter++;
                     }
                     else
                     {
                        P_offd_data_new[new_offd_pos] = value;
                        P_offd_j_big[new_offd_pos] = aux_j[j];
                        new_offd_pos++;
                        is_q[new_j_counter] = is_q[j];
                        new_j_counter++;
                     }

                  }
               }/* end element loop */


               /* adjust p_count and j_pos */
               p_count_diag -= lost_counter_diag;
               p_count_offd -= lost_counter_offd;

               j_diag_pos -= lost_counter_diag;
               j_offd_pos -= lost_counter_offd;


            } /* end of num lost > 0 */

         }/* end of q_max > 0 */


      }/* end of TRUNCATION */

      /* modify i */
      orig_diag_start = P_diag_i[i + 1];
      orig_offd_start = P_offd_i[i + 1];

      P_diag_i_new[i + 1] = P_diag_i_new[i] + p_count_diag;
      P_offd_i_new[i + 1] = P_offd_i_new[i] + p_count_offd;


      if (j_diag_pos != P_diag_i_new[i + 1])
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Warning - diag Row Problem in hypre_BoomerAMG_GMExpandInterp!\n");
      }
      if (j_offd_pos != P_offd_i_new[i + 1])
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Warning - off-diag Row Problem in hypre_BoomerAMG_GMExpandInterp!\n");

      }

   } /* END loop through rows of P */


   /* Done looping through rows - NOW FINISH THINGS UP! */

   /* if level = first_level , we need to update the number of
   * funcs and the dof_func */

   if (level == interp_vec_first_level )
   {
      HYPRE_Int spot;

      c_dof_func = hypre_TReAlloc_v2(c_dof_func,  HYPRE_Int, hypre_IntArraySize(*coarse_dof_func),
                                     HYPRE_Int,  new_ncv, hypre_IntArrayMemoryLocation(*coarse_dof_func));
      spot = 0;

      for (i = 0; i < ncv_peru; i++)
      {
         for (k = 0; k < num_functions + num_smooth_vecs; k++)
         {
            c_dof_func[spot++] = k;
         }
      }

      /*RETURN: update num functions  and dof_func */
      new_nf =  num_functions + num_smooth_vecs;

      *nf = new_nf;
      hypre_IntArrayData(*coarse_dof_func) = c_dof_func;
      hypre_IntArraySize(*coarse_dof_func) = new_ncv;


      /* also we need to update the col starts and global num columns*/

      /* assumes that unknowns are together on a procsessor with
       * nodal coarsening  */
      new_col_starts[0] = (col_starts[0] / (HYPRE_BigInt)num_functions) * (HYPRE_BigInt)new_nf ;
      new_col_starts[1] = (col_starts[1] / (HYPRE_BigInt)num_functions) * (HYPRE_BigInt)new_nf;

      if (myid == (num_procs - 1)) { g_nc = new_col_starts[1]; }
      hypre_MPI_Bcast(&g_nc, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else /* not first level */
   {
      /* grab global num cols */
      g_nc = hypre_ParCSRMatrixGlobalNumCols(*P);

      /* copy col starts */
      new_col_starts[0] = col_starts[0];
      new_col_starts[1] = col_starts[1];
   }


   /* modify P - now P has more entries and possibly more cols -
    * need to create a new P and destory old*/

   new_P = hypre_ParCSRMatrixCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    g_nc,
                                    hypre_ParCSRMatrixColStarts(A),
                                    new_col_starts,
                                    0,
                                    P_diag_i_new[nv],
                                    P_offd_i_new[nv]);


   P_diag = hypre_ParCSRMatrixDiag(new_P);
   hypre_CSRMatrixI(P_diag) = P_diag_i_new;
   hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
   hypre_CSRMatrixData(P_diag) = P_diag_data_new;
   hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_i_new[num_rows_P];

   P_offd = hypre_ParCSRMatrixOffd(new_P);
   hypre_CSRMatrixData(P_offd) = P_offd_data_new;
   hypre_CSRMatrixI(P_offd) = P_offd_i_new;

   /* If parallel we need to do the col map offd! */
   if (num_procs > 1)
   {
      HYPRE_Int count;
      HYPRE_Int num_cols_P_offd = 0;
      HYPRE_Int P_offd_new_size = P_offd_i_new[num_rows_P];

      if (P_offd_new_size)
      {

         HYPRE_BigInt *j_copy;

         /* check this */
         new_col_map_offd_P = hypre_CTAlloc(HYPRE_BigInt,  P_offd_new_size, HYPRE_MEMORY_HOST);

         /*first copy the j entries (these are GLOBAL numbers) */
         j_copy = hypre_CTAlloc(HYPRE_BigInt,  P_offd_new_size, HYPRE_MEMORY_HOST);
         for (i = 0; i < P_offd_new_size; i++)
         {
            j_copy[i] = P_offd_j_big[i];
         }

         /* now sort them */
         hypre_BigQsort0(j_copy, 0, P_offd_new_size - 1);

         /* now copy to col_map offd - but only each col once */
         new_col_map_offd_P[0] = j_copy[0];
         count = 0;
         for (i = 0; i < P_offd_new_size; i++)
         {
            if (j_copy[i] > new_col_map_offd_P[count])
            {
               count++;
               new_col_map_offd_P[count] = j_copy[i];
            }
         }
         num_cols_P_offd = count + 1;

         /* reset the j entries to be local */
         for (i = 0; i < P_offd_new_size; i++)
            P_offd_j_new[i] = hypre_BigBinarySearch(new_col_map_offd_P,
                                                    P_offd_j_big[i],
                                                    num_cols_P_offd);
         hypre_TFree(j_copy, HYPRE_MEMORY_HOST);
      }

      hypre_ParCSRMatrixColMapOffd(new_P) = new_col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;

   } /* end col map stuff */

   hypre_CSRMatrixJ(P_offd) =  P_offd_j_new;

   /* CREATE THE COMM PKG */
   hypre_MatvecCommPkgCreate ( new_P );


#if SV_DEBUG
   {
      char new_file[80];
      hypre_CSRMatrix *P_CSR;

      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(new_P);

      if (!myid)
      {
         hypre_sprintf(new_file, "%s.level.%d", "P_new_new", level );
         if (P_CSR)
         {
            hypre_CSRMatrixPrint(P_CSR, new_file);
         }
      }

      hypre_CSRMatrixDestroy(P_CSR);
   }

#endif

   /*destroy old */
   hypre_ParCSRMatrixDestroy(*P);

   /* RETURN: update P */
   *P = new_P;

   /* clean up */
   hypre_TFree(is_q, HYPRE_MEMORY_HOST);
   hypre_TFree(q_data, HYPRE_MEMORY_HOST);
   hypre_TFree(q_trunc_data, HYPRE_MEMORY_HOST);
   hypre_TFree(aux_j, HYPRE_MEMORY_HOST);
   hypre_TFree(aux_data, HYPRE_MEMORY_HOST);
   hypre_TFree(is_diag, HYPRE_MEMORY_HOST);
   hypre_TFree(P_offd_j_big, HYPRE_MEMORY_HOST);

   hypre_TFree(q_count_sv, HYPRE_MEMORY_HOST);
   hypre_TFree(num_lost_sv, HYPRE_MEMORY_HOST);
   hypre_TFree(lost_value_sv, HYPRE_MEMORY_HOST);
   hypre_TFree(lost_counter_q_sv, HYPRE_MEMORY_HOST);
   hypre_TFree(q_dist_value_sv, HYPRE_MEMORY_HOST);
   hypre_TFree(col_map, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_to_fine, HYPRE_MEMORY_HOST);

   hypre_TFree(smooth_vec_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/******************************************************************************
  hypre_BoomerAMGRefineInterp-

* this is an update to the current P - a.k.a. "iterative weight
  interpolation"

******************************************************************************/

HYPRE_Int
hypre_BoomerAMGRefineInterp( hypre_ParCSRMatrix *A,
                             hypre_ParCSRMatrix *P,
                             HYPRE_BigInt *num_cpts_global,
                             HYPRE_Int *nf,
                             HYPRE_Int *dof_func,
                             HYPRE_Int *CF_marker,
                             HYPRE_Int level)
{
   HYPRE_UNUSED_VAR(level);

   HYPRE_Int        i, j, k, pp;
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int       *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int        num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);
   HYPRE_Real      *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int       *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int       *P_diag_j = hypre_CSRMatrixJ(P_diag);
   HYPRE_Int        num_rows_P = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int        P_diag_size = P_diag_i[num_rows_P];

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int       *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int        P_offd_size = P_offd_i[num_rows_P];

   HYPRE_Real      *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int       *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Int        num_cols_P_offd = hypre_CSRMatrixNumCols(P_offd);
   HYPRE_BigInt    *col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(P);

   HYPRE_Int orig_diag_start, orig_offd_start;
   HYPRE_Int j_diag_pos, j_offd_pos;
   HYPRE_Int fcn_num, p_num_diag_elements, p_num_offd_elements;

   HYPRE_Real *P_diag_data_new;
   HYPRE_Real *P_offd_data_new;

   HYPRE_Int  *CF_marker_offd = NULL;
   HYPRE_Int  *dof_func_offd = NULL;

   HYPRE_BigInt  *fine_to_coarse_offd;

   HYPRE_Int found;

   HYPRE_Int num_functions = *nf;


   hypre_ParCSRCommPkg     *comm_pkg_P = hypre_ParCSRMatrixCommPkg(P);
   hypre_ParCSRCommPkg     *comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);

   MPI_Comm             comm;


   HYPRE_Int      coarse_counter;
   HYPRE_Int      j_ext_index;


   HYPRE_Int      *fine_to_coarse;
   HYPRE_Int       k_point, j_point, j_point_c, p_point;
   HYPRE_BigInt    big_k, big_index, big_j_point_c;

   HYPRE_Real      diagonal, aw, a_ij;
   HYPRE_Int       scale_row;
   HYPRE_Real      sum;

   HYPRE_Real      new_row_sum, orig_row_sum;
   HYPRE_Int       use_alt_w, kk, kk_count, cur_spot;
   HYPRE_Int       dist_coarse;

   hypre_CSRMatrix *P_ext = NULL;
   HYPRE_Real      *P_ext_data = NULL;
   HYPRE_Int       *P_ext_i = NULL;
   HYPRE_BigInt    *P_ext_j = NULL;

   HYPRE_Int        num_sends_A, index, start;
   HYPRE_Int        myid = 0, num_procs = 1;


   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int       *int_buf_data = NULL;
   HYPRE_BigInt    *big_buf_data = NULL;

   if (!comm_pkg_P)
   {
      hypre_MatvecCommPkgCreate (P);
      comm_pkg_P = hypre_ParCSRMatrixCommPkg(P);
   }
   comm = hypre_ParCSRCommPkgComm(comm_pkg_A);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

#if SV_DEBUG
   {
      char new_file[80];

      hypre_CSRMatrix *P_CSR = NULL;

      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(P);
      if (!myid)
      {
         hypre_sprintf(new_file, "%s.level.%d", "P_new_orig", level );
         if (P_CSR)
         {
            hypre_CSRMatrixPrint(P_CSR, new_file);
         }
      }

      hypre_CSRMatrixDestroy(P_CSR);


      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A);
      if (!myid)
      {
         hypre_sprintf(new_file, "%s.level.%d", "A", level );
         if (P_CSR)
         {
            hypre_CSRMatrixPrint(P_CSR, new_file);
         }
      }

      hypre_CSRMatrixDestroy(P_CSR);

   }

#endif


   num_sends_A = hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   big_buf_data = hypre_CTAlloc(HYPRE_BigInt, hypre_ParCSRCommPkgSendMapStart(comm_pkg_A,
                                                                              num_sends_A), HYPRE_MEMORY_HOST);
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg_A,
                                                                           num_sends_A), HYPRE_MEMORY_HOST);


   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/
   {
      HYPRE_BigInt my_first_cpt;
      HYPRE_Int tmp_i;

      my_first_cpt = num_cpts_global[0];

      /* need a fine-to-coarse mapping (num row P = num rows A)*/
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  num_rows_P, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_rows_P; i++) { fine_to_coarse[i] = -1; }

      coarse_counter = 0;
      for (i = 0; i < num_rows_P; i++)
      {
         if (CF_marker[i] >= 0)
         {
            fine_to_coarse[i] = coarse_counter;
            coarse_counter++;
         }
      }

      /* now from other procs */
      fine_to_coarse_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_A_offd, HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < num_sends_A; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i + 1); j++)
         {

            tmp_i = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A, j)];
            big_buf_data[index++] = (HYPRE_BigInt)tmp_i + my_first_cpt; /* makes it global */
         }

      }

      comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg_A, big_buf_data,
                                                  fine_to_coarse_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);

   }


   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns of A
    *-------------------------------------------------------------------*/
   {

      if (num_cols_A_offd)
      {
         CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
      }

      if (num_functions > 1 && num_cols_A_offd)
      {
         dof_func_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
      }

      index = 0;
      for (i = 0; i < num_sends_A; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i + 1); j++)
         {
            int_buf_data[index++] = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A, j)];
         }

      }

      comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg_A, int_buf_data,
                                                  CF_marker_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
      if (num_functions > 1)
      {
         index = 0;
         for (i = 0; i < num_sends_A; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i + 1); j++)
            {
               int_buf_data[index++]
                  = dof_func[hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A, j)];
            }

         }

         comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg_A, int_buf_data,
                                                     dof_func_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

   }


   /*-------------------------------------------------------------------
    * Get the ghost rows of P
    *-------------------------------------------------------------------*/
   {

      HYPRE_Int kc;
      HYPRE_BigInt col_1 = hypre_ParCSRMatrixFirstColDiag(P);
      HYPRE_BigInt col_n = col_1 + hypre_CSRMatrixNumCols(P_diag);

      if (num_procs > 1)
      {
         /* need the rows of P on other processors associated with
            the offd cols of A */
         P_ext      = hypre_ParCSRMatrixExtractBExt(P, A, 1);
         P_ext_i    = hypre_CSRMatrixI(P_ext);
         P_ext_j    = hypre_CSRMatrixBigJ(P_ext);
         P_ext_data = hypre_CSRMatrixData(P_ext);
      }

      index = 0;
      /* now check whether each col is in the diag of offd part of P)*/
      for (i = 0; i < num_cols_A_offd; i++)
      {
         for (j = P_ext_i[i]; j < P_ext_i[i + 1]; j++)
         {
            big_k = P_ext_j[j];
            /* is it in the diag ?*/
            if (big_k >= col_1 && big_k < col_n)
            {
               P_ext_j[index] = big_k - col_1;  /* make a local col number */
               P_ext_data[index++] = P_ext_data[j];
            }
            else
            {
               /* off diag entry */
               kc = hypre_BigBinarySearch(col_map_offd_P, big_k, num_cols_P_offd);
               /* now this corresponds to the location in the col_map_offd
                ( so it is a local column number */
               if (kc > -1)
               {
                  P_ext_j[index] = (HYPRE_BigInt)(-kc - 1); /* make negative */
                  P_ext_data[index++] = P_ext_data[j];
               }
            }
         }
         P_ext_i[i] = index;
      }
      for (i = num_cols_A_offd; i > 0; i--)
      {
         P_ext_i[i] = P_ext_i[i - 1];
      }

      if (num_procs > 1) { P_ext_i[0] = 0; }


   } /* end of ghost rows */


   /* initialized to zero */
   P_diag_data_new = hypre_CTAlloc(HYPRE_Real, P_diag_size, memory_location);
   P_offd_data_new = hypre_CTAlloc(HYPRE_Real, P_offd_size, memory_location);

   j_diag_pos = 0;
   j_offd_pos = 0;

   /*-------------------------------------------------------------------
    *loop through rows
    *-------------------------------------------------------------------*/
   for (i = 0; i < num_rows_P; i++)
   {
      new_row_sum = 0.0;
      use_alt_w = 0;
      scale_row = 0;
      orig_row_sum = 0.0;

      fcn_num = (HYPRE_Int) fmod(i, num_functions);
      if (fcn_num != dof_func[i])
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "WARNING - ROWS incorrectly ordered in hypre_BoomerAMGRefineInterp!\n");
      }

      /* number of elements in row of p*/
      orig_diag_start =  P_diag_i[i];
      orig_offd_start =  P_offd_i[i];

      /* number of elements in row */
      p_num_diag_elements = P_diag_i[i + 1] - orig_diag_start;
      p_num_offd_elements = P_offd_i[i + 1] - orig_offd_start;

      if (CF_marker[i] >= 0) /* row corres. to coarse point - just copy orig */
      {
         /* diag */
         for (j = 0; j < p_num_diag_elements; j++)
         {
            P_diag_data_new[j_diag_pos++] = P_diag_data[orig_diag_start + j];
         }
         /*offd */
         for (j = 0; j < p_num_offd_elements; j++)
         {
            P_offd_data_new[j_offd_pos++] = P_offd_data[orig_offd_start + j];
         }
      }
      else /* row is for fine point  - make new interpolation*/
      {
         /* make orig entries zero*/
         for (j = 0; j < p_num_diag_elements; j++)
         {
            orig_row_sum +=  P_diag_data[orig_diag_start + j];
            P_diag_data_new[j_diag_pos++] = 0.0;
         }
         for (j = 0; j < p_num_offd_elements; j++)
         {
            orig_row_sum +=  P_offd_data[orig_offd_start + j];
            P_offd_data_new[j_offd_pos++] = 0.0;
         }

         /*get diagonal of A */
         diagonal = A_diag_data[A_diag_i[i]];

         /* loop over elements in row i of A (except diagonal element)*/
         /* diag*/
         for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
         {
            j_point = A_diag_j[j];

            /* only want like unknowns */
            if (fcn_num != dof_func[j_point])
            {
               continue;
            }

            dist_coarse = 0;
            a_ij = A_diag_data[j];

            found = 0;
            if (CF_marker[j_point] >= 0) /*coarse*/
            {
               j_point_c = fine_to_coarse[j_point];

               /* find P(i,j_c) and put value there (there may not be
                  an entry in P if this coarse connection was not a
                  strong connection */

               /* we are looping in the diag of this row, so we only
                * need to look in P_diag */
               for (k = P_diag_i[i]; k < P_diag_i[i + 1]; k ++)
               {
                  if (P_diag_j[k] == j_point_c)
                  {
                     P_diag_data_new[k] += a_ij;
                     found = 1;
                     break;
                  }
               }
               if (!found)
               {
                  /*this is a weakly connected c-point - does
                    not contribute - so no error - but this messes up row sum*/
                  /* we need to distribute this */
                  dist_coarse = 1;
               }
            }
            else /*fine connection  */
            {

               sum = 0.0;

               /*loop over diag and offd of row of P for j_point and
                 get the sum of the connections to c-points of i
                 (diag and offd)*/
               /*diag*/
               for (pp = P_diag_i[j_point]; pp < P_diag_i[j_point + 1]; pp++)
               {
                  p_point = P_diag_j[pp];/* this is a coarse index */
                  /* is p_point in row i also ?  check the diag part*/
                  for (k = P_diag_i[i]; k < P_diag_i[i + 1]; k ++)
                  {
                     k_point = P_diag_j[k]; /* this is a coarse index */
                     if (p_point == k_point)
                     {
                        /* add p_jk to sum */
                        sum += P_diag_data[pp];

                        break;
                     }
                  }/* end loop k over row i */

               } /* end loop pp over row j_point for diag */
               /* now offd */
               for (pp = P_offd_i[j_point]; pp < P_offd_i[j_point + 1]; pp++)
               {
                  p_point = P_offd_j[pp];/* this is a coarse index */
                  /* is p_point in row i also ? check the offd part*/
                  for (k = P_offd_i[i]; k < P_offd_i[i + 1]; k ++)
                  {
                     k_point = P_offd_j[k]; /* this is a coarse index */
                     if (p_point == k_point)
                     {
                        /* add p_jk to sum */
                        sum += P_offd_data[pp];

                        break;
                     }
                  }/* end loop k over row i */

               } /* end loop pp over row j_point */

               if (hypre_abs(sum) < 1e-12)
               {
                  sum = 1.0;
                  use_alt_w = 1;
               }

               if (use_alt_w)
               {
                  /* distribute a_ij equally among coarse points */
                  aw =  a_ij / (p_num_diag_elements + p_num_offd_elements);
                  kk_count = 0;
                  /* loop through row i of orig p*/
                  /* diag */
                  for (kk = P_diag_i[i]; kk < P_diag_i[i + 1]; kk++)
                  {
                     cur_spot =  P_diag_i[i] + kk_count;
                     P_diag_data_new[cur_spot] += aw;

                     kk_count++;
                  }
                  /* offd */
                  kk_count = 0;
                  for (kk = P_offd_i[i]; kk < P_offd_i[i + 1]; kk++)
                  {
                     cur_spot =  P_offd_i[i] + kk_count;
                     P_offd_data_new[cur_spot] += aw;

                     kk_count++;
                  }
                  /* did each element of p */

                  /* skip out to next jj of A */
                  continue;

               }/* end of alt w */

               /* Now we need to do the distributing  */

               /* loop through row i (diag and offd )of p*/
               /* first diag part */
               for (k = P_diag_i[i]; k < P_diag_i[i + 1]; k ++)
               {
                  k_point = P_diag_j[k]; /* this is a coarse index */
                  /* now is there an entry for P(j_point, k_point)?
                   - need to look through row j_point (on -proc since
                   j came from A_diag */
                  for (pp = P_diag_i[j_point]; pp < P_diag_i[j_point + 1]; pp++)
                  {
                     if (P_diag_j[pp] == k_point)
                     {
                        /* a_ij*w_jk */
                        aw =  a_ij * P_diag_data[pp];
                        aw = aw / sum;

                        P_diag_data_new[k] += aw;
                        break;
                     }
                  } /* end loop pp over row j_point */
               } /* end loop k over diag row i of P */
               for (k = P_offd_i[i]; k < P_offd_i[i + 1]; k ++)
               {
                  k_point = P_offd_j[k]; /* this is a coarse index */
                  /* now is there an entry for P(j_point, k_point)?
                   - need to look through offd part of row j_point
                   (this is on -proc since j came from A_diag */
                  for (pp = P_offd_i[j_point]; pp < P_offd_i[j_point + 1]; pp++)
                  {
                     if (P_offd_j[pp] == k_point)
                     {
                        /* a_ij*w_jk */
                        aw =  a_ij * P_offd_data[pp];
                        aw = aw / sum;

                        P_offd_data_new[k] += aw;
                        break;
                     }
                  } /* end loop pp over row j_point */
               } /* end loop k over row i of P */

            } /* end of fine connection in row of A*/

            if (dist_coarse)
            {
               /* coarse not in orig interp.(weakly connected) */
               /* distribute a_ij equally among coarse points */
               aw =  a_ij / (p_num_diag_elements + p_num_offd_elements);
               kk_count = 0;
               /* loop through row i of orig p*/
               for (kk = P_diag_i[i]; kk < P_diag_i[i + 1]; kk++)
               {
                  cur_spot =  P_diag_i[i] + kk_count;
                  P_diag_data_new[cur_spot] += aw;

                  kk_count++;
               }
               kk_count = 0;
               for (kk = P_offd_i[i]; kk < P_offd_i[i + 1]; kk++)
               {
                  cur_spot =  P_offd_i[i] + kk_count;
                  P_offd_data_new[cur_spot] += aw;

                  kk_count++;
               }
            }

         }/* end loop j over row i of A_diag */

         /* loop over offd of A */

         /* loop over elements in row i of A_offd )*/
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            j_point = A_offd_j[j];

            /* only want like unknowns  - check the offd dof func*/
            if (fcn_num != dof_func_offd[j_point])
            {
               continue;
            }

            dist_coarse = 0;
            a_ij = A_offd_data[j];

            found = 0;

            if (CF_marker_offd[j_point] >= 0) /*check the offd marker*/
            {
               /* coarse */
               big_j_point_c = fine_to_coarse_offd[j_point]; /* now its global!! */

               /* find P(i,j_c) and put value there (there may not be
                  an entry in P if this coarse connection was not a
                  strong connection */

               /* we are looping in the off diag of this row, so we only
                * need to look in P_offd */
               for (k = P_offd_i[i]; k < P_offd_i[i + 1]; k ++)
               {
                  index = P_offd_j[k]; /* local number */
                  big_index = col_map_offd_P[index]; /*global number
                                                   * (becuz j_point_c
                                                   * is global */


                  /* if (P_offd_j[k] == j_point_c)*/
                  if (big_index == big_j_point_c)
                  {
                     P_offd_data_new[k] += a_ij;
                     found = 1;
                     break;
                  }
               }
               if (!found)
               {
                  /*this is a weakly connected c-point - does
                    not contribute - so no error - but this messes up row sum*/
                  /* we need to distribute this */
                  dist_coarse = 1;
               }
            }
            else /*fine connection  */
            {

               sum = 0.0;

               /*loop over row of P for j_point and get the sum of
                 the connections to c-points of i (diag and offd) -
                 now the row for j_point is on another processor -
                 and j_point is an index of A - need to convert it to
                 corresponding index of P */

               /* j_point is an index of A_off d - so */
               /* now this is the row in P, but these are stored in
                * P_ext according to offd of A */
               j_ext_index = j_point;

               for (pp = P_ext_i[j_ext_index]; pp < P_ext_i[j_ext_index + 1]; pp++)
               {
                  p_point = (HYPRE_Int)P_ext_j[pp];/* this is a coarse index */
                  /* is p_point in row i of P also ?  check the diag and
                     offd part or row i of P */

                  if (p_point > -1) /* in diag part */
                  {
                     for (k = P_diag_i[i]; k < P_diag_i[i + 1]; k ++)
                     {
                        k_point = P_diag_j[k]; /* this is a coarse index */
                        if (p_point == k_point)
                        {
                           /* add p_jk to sum */
                           sum += P_ext_data[pp];

                           break;
                        }
                     }/* end loop k over row i */
                  }
                  else /* in offd diag part */
                  {
                     p_point = -p_point - 1;
                     /* p_point is a local col number for P now */
                     for (k = P_offd_i[i]; k < P_offd_i[i + 1]; k ++)
                     {
                        k_point = P_offd_j[k]; /* this is a coarse index */
                        if (p_point == k_point)
                        {
                           /* add p_jk to sum */
                           sum += P_ext_data[pp];

                           break;
                        }
                     }/* end loop k over row i */
                  }/* end diag or offd */
               }/* end loop over row P for j_point */

               if (hypre_abs(sum) < 1e-12)
               {
                  sum = 1.0;
                  use_alt_w = 1;
               }

               if (use_alt_w)
               {
                  /* distribute a_ij equally among coarse points */
                  aw =  a_ij / (p_num_diag_elements + p_num_offd_elements);
                  kk_count = 0;
                  /* loop through row i of orig p*/
                  /* diag */
                  for (kk = P_diag_i[i]; kk < P_diag_i[i + 1]; kk++)
                  {
                     cur_spot =  P_diag_i[i] + kk_count;
                     P_diag_data_new[cur_spot] += aw;

                     kk_count++;
                  }
                  /* offd */
                  kk_count = 0;
                  for (kk = P_offd_i[i]; kk < P_offd_i[i + 1]; kk++)
                  {
                     cur_spot =  P_offd_i[i] + kk_count;
                     P_offd_data_new[cur_spot] += aw;

                     kk_count++;
                  }
                  /* did each element of p */

                  /* skip out to next jj of A */
                  continue;

               }/* end of alt w */

               /* Now we need to do the distributing  */

               /* loop through row i (diag and offd )of p*/
               /* first diag part */
               for (k = P_diag_i[i]; k < P_diag_i[i + 1]; k ++)
               {
                  k_point = P_diag_j[k]; /* this is a coarse index */
                  /* now is there an entry for P(j_point, k_point)?
                     - need to look through row j_point  - this will
                     be off-proc */

                  for (pp = P_ext_i[j_ext_index]; pp < P_ext_i[j_ext_index + 1]; pp++)
                  {
                     p_point = (HYPRE_Int) P_ext_j[pp];
                     if (p_point > -1) /* diag part */
                     {
                        if (p_point == k_point)
                        {
                           /* a_ij*w_jk */
                           aw =  a_ij * P_ext_data[pp];
                           aw = aw / sum;

                           P_diag_data_new[k] += aw;
                           break;
                        }
                     }

                  } /* end loop pp over row j_point */
               } /* end loop k over diag row i of P */
               for (k = P_offd_i[i]; k < P_offd_i[i + 1]; k ++)
               {
                  k_point = P_offd_j[k]; /* this is a coarse index */
                  /* now is there an entry for P(j_point, k_point)?
                    - need to look through row j_point  - this will
                     be off-proc */
                  for (pp = P_ext_i[j_ext_index]; pp < P_ext_i[j_ext_index + 1]; pp++)
                  {
                     p_point = (HYPRE_Int) P_ext_j[pp];
                     if (p_point < 0) /* in offd part */
                     {
                        p_point = - p_point - 1;
                        if (p_point == k_point)
                        {
                           /* a_ij*w_jk */
                           aw =  a_ij * P_ext_data[pp];
                           aw = aw / sum;

                           P_offd_data_new[k] += aw;
                           break;
                        }
                     }

                  } /* end loop pp over row j_point */

               } /* end loop k over row i of P */

            } /* end of fine connection in row of A*/

            if (dist_coarse)
            {
               /* coarse not in orig interp.(weakly connected) */
               /* distribute a_ij equally among coarse points */
               aw =  a_ij / (p_num_diag_elements + p_num_offd_elements);
               kk_count = 0;
               /* loop through row i of orig p*/
               for (kk = P_diag_i[i]; kk < P_diag_i[i + 1]; kk++)
               {
                  cur_spot =  P_diag_i[i] + kk_count;
                  P_diag_data_new[cur_spot] += aw;

                  kk_count++;
               }
               kk_count = 0;
               for (kk = P_offd_i[i]; kk < P_offd_i[i + 1]; kk++)
               {
                  cur_spot =  P_offd_i[i] + kk_count;
                  P_offd_data_new[cur_spot] += aw;

                  kk_count++;
               }
            }

         }/* end loop j over row i of A_offd */

         /* now divide by the diagonal and we are finished with this row!*/
         if (hypre_abs(diagonal) > 0.0)
         {
            for (k = P_diag_i[i] ; k < P_diag_i[i + 1]; k++)
            {
               P_diag_data_new[k] /= -(diagonal);
               new_row_sum +=  P_diag_data_new[k];

            }
            for (k = P_offd_i[i] ; k < P_offd_i[i + 1]; k++)
            {
               P_offd_data_new[k] /= -(diagonal);
               new_row_sum +=  P_offd_data_new[k];

            }

            /* now re-scale */
            if (scale_row)
            {

               for (k = P_diag_i[i] ; k < P_diag_i[i + 1]; k++)
               {
                  P_diag_data_new[k] *= (orig_row_sum / new_row_sum);

               }
               for (k = P_offd_i[i] ; k < P_offd_i[i + 1]; k++)
               {
                  P_offd_data_new[k] *= (orig_row_sum / new_row_sum);

               }


            }

         }

      } /* end of row of P is fine point - build interp */

   } /* end of i loop throw rows */

   /* modify P - only need to replace the data (i and j are the same)*/
   hypre_TFree(P_diag_data, memory_location);
   hypre_TFree(P_offd_data, memory_location);

   hypre_CSRMatrixData(P_diag) = P_diag_data_new;
   hypre_CSRMatrixData(P_offd) = P_offd_data_new;


#if SV_DEBUG
   {
      char new_file[80];
      hypre_CSRMatrix *P_CSR;

      P_CSR = hypre_ParCSRMatrixToCSRMatrixAll(P);

      if (!myid)
      {
         hypre_sprintf(new_file, "%s.level.%d", "P_new_new", level );
         if (P_CSR)
         {
            hypre_CSRMatrixPrint(P_CSR, new_file);
         }
      }

      hypre_CSRMatrixDestroy(P_CSR);
   }

#endif

   /* clean up */
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(dof_func_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(big_buf_data, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixDestroy(P_ext);

   return hypre_error_flag;
}
