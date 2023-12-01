/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*====================
 * Functions to run cr
 *====================*/
#include <_hypre_parcsr_ls.h>

#define RelaxScheme1 3 /* cr type */
#define fptOmegaJac 1  /* 1 is f pt weighted jacobi */
#define omega1 1.0     /* weight */
#define fptgs 3        /* 3 is f pt GS */

#define theta_global1 .7    /* cr stop criteria */
#define mu1            5    /* # of cr sweeps */

#define cpt  1
#define fpt -1
#define cand 0

HYPRE_Int
hypre_BoomerAMGCoarsenCR1( hypre_ParCSRMatrix    *A,
                           hypre_IntArray       **CF_marker_ptr,
                           HYPRE_BigInt          *coarse_size_ptr,
                           HYPRE_Int              num_CR_relax_steps,
                           HYPRE_Int              IS_type,
                           HYPRE_Int              CRaddCpoints)
{
   HYPRE_UNUSED_VAR(num_CR_relax_steps);
   HYPRE_UNUSED_VAR(IS_type);

   HYPRE_Int i;
   /* HYPRE_Real theta_global;*/
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int       *A_i           = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_j           = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real      *A_data        = hypre_CSRMatrixData(A_diag);
   HYPRE_Int        num_variables = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int       *CF_marker;
   HYPRE_Int        coarse_size;

   if (CRaddCpoints == 0)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_variables);
      hypre_IntArrayInitialize(*CF_marker_ptr);
      hypre_IntArraySetConstantValues(*CF_marker_ptr, fpt);
   }
   CF_marker = hypre_IntArrayData(*CF_marker_ptr);

   /* Run the CR routine */

   hypre_fprintf(stdout, "\n... Building CF using CR ...\n\n");
   hypre_cr(A_i, A_j, A_data, num_variables, CF_marker,
            RelaxScheme1, omega1, theta_global1, mu1);

   hypre_fprintf(stdout, "\n... Done \n\n");
   coarse_size = 0;
   for ( i = 0 ; i < num_variables; i++)
   {
      if ( CF_marker[i] == cpt)
      {
         coarse_size++;
      }
   }
   *coarse_size_ptr = coarse_size;

   return hypre_error_flag;
}

/* main cr routine */
HYPRE_Int hypre_cr(HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data, HYPRE_Int n, HYPRE_Int *cf,
                   HYPRE_Int rlx, HYPRE_Real omega, HYPRE_Real tg, HYPRE_Int mu)
{
   HYPRE_Int i, nstages = 0;
   HYPRE_Real rho, rho0, rho1, *e0, *e1;
   HYPRE_Real nc = 0.0;

   e0 = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   e1 = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);

   hypre_fprintf(stdout, "Stage  \t rho \t alpha \n");
   hypre_fprintf(stdout, "-----------------------\n");

   for (i = 0; i < n; i++)
   {
      e1[i] = 1.0e0 + .1 * hypre_RandI();
   }

   /* stages */
   while (1)
   {
      if (nstages > 0)
      {
         for (i = 0; i < n; i++)
         {
            if (cf[i] == cpt)
            {
               e0[i] = 0.0e0;
               e1[i] = 0.0e0;
            }
         }
      }

      switch (rlx)
      {
         case fptOmegaJac:
            for (i = 0; i < mu; i++)
            {
               hypre_fptjaccr(cf, A_i, A_j, A_data, n, e0, omega, e1);
            }
            break;
         case fptgs:
            for (i = 0; i < mu; i++)
            {
               hypre_fptgscr(cf, A_i, A_j, A_data, n, e0, e1);
            }
            break;
      }

      rho = 0.0e0; rho0 = 0.0e0; rho1 = 0.0e0;
      for (i = 0; i < n; i++)
      {
         rho0 += hypre_pow(e0[i], 2);
         rho1 += hypre_pow(e1[i], 2);
      }
      rho = hypre_sqrt(rho1) / hypre_sqrt(rho0);

      if (rho > tg)
      {
         hypre_formu(cf, n, e1, A_i, rho);
         hypre_IndepSetGreedy(A_i, A_j, n, cf);

         hypre_fprintf(stdout, "  %d \t%2.3f  \t%2.3f \n",
                       nstages, rho, nc / n);
         /* update for next sweep */
         nc = 0.0e0;
         for (i = 0; i < n; i++)
         {
            if (cf[i] ==  cpt)
            {
               nc += 1.0e0;
            }
            else if (cf[i] ==  fpt)
            {
               e0[i] = 1.0e0 + .1 * hypre_RandI();
               e1[i] = 1.0e0 + .1 * hypre_RandI();
            }
         }
         nstages += 1;
      }
      else
      {
         hypre_fprintf(stdout, "  %d \t%2.3f  \t%2.3f \n",
                       nstages, rho, nc / n);
         break;
      }
   }

   hypre_TFree(e0, HYPRE_MEMORY_HOST);
   hypre_TFree(e1, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* take an ind. set over the candidates*/
HYPRE_Int hypre_GraphAdd( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index,
                          HYPRE_Int istack )
{
   HYPRE_Int prev = tail[-istack];

   list[index].prev = prev;
   if (prev < 0)
   {
      head[-istack] = index;
   }
   else
   {
      list[prev].next = index;
   }
   list[index].next = -istack;
   tail[-istack] = index;

   return hypre_error_flag;
}

HYPRE_Int hypre_GraphRemove( Link *list, HYPRE_Int *head, HYPRE_Int *tail, HYPRE_Int index )
{
   HYPRE_Int prev = list[index].prev;
   HYPRE_Int next = list[index].next;

   if (prev < 0)
   {
      head[prev] = next;
   }
   else
   {
      list[prev].next = next;
   }
   if (next < 0)
   {
      tail[next] = prev;
   }
   else
   {
      list[next].prev = prev;
   }

   return hypre_error_flag;
}

HYPRE_Int hypre_IndepSetGreedy(HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf)
{
   Link *list;
   HYPRE_Int  *head, *head_mem, *ma;
   HYPRE_Int  *tail, *tail_mem;

   HYPRE_Int i, ji, jj, jl, index, istack, stack_size;

   ma = hypre_CTAlloc(HYPRE_Int,  n, HYPRE_MEMORY_HOST);

   /* Initialize the graph and measure array
    *
    * ma: cands >= 1
    *     cpts  = -1
    *     else  =  0
    * Note: only cands are put into graph */

   istack = 0;
   for (i = 0; i < n; i++)
   {
      if (cf[i] == cand)
      {
         ma[i] = 1;
         for (ji = A_i[i] + 1; ji < A_i[i + 1]; ji++)
         {
            jj = A_j[ji];
            if (cf[jj] != cpt)
            {
               ma[i]++;
            }
         }
         if (ma[i] > istack)
         {
            istack = (HYPRE_Int) ma[i];
         }
      }
      else if (cf[i] == cpt)
      {
         ma[i] = -1;
      }
      else
      {
         ma[i] = 0;
      }
   }
   stack_size = 2 * istack;

   /* initialize graph */
   head_mem = hypre_CTAlloc(HYPRE_Int,  stack_size, HYPRE_MEMORY_HOST); head = head_mem + stack_size;
   tail_mem = hypre_CTAlloc(HYPRE_Int,  stack_size, HYPRE_MEMORY_HOST); tail = tail_mem + stack_size;
   list = hypre_CTAlloc(Link,  n, HYPRE_MEMORY_HOST);

   for (i = -1; i >= -stack_size; i--)
   {
      head[i] = i;
      tail[i] = i;
   }
   for (i = 0; i < n; i++)
   {
      if (ma[i] > 0)
      {
         hypre_GraphAdd(list, head, tail, i, (HYPRE_Int) ma[i]);
      }
   }

   /* Loop until all points are either F or C */
   while (istack > 0)
   {
      /* i w/ max measure at head of stacks */
      i = head[-istack];

      /* make i C point */
      cf[i] = cpt;
      ma[i] = -1;

      /* remove i from graph */
      hypre_GraphRemove(list, head, tail, i);

      /* update nbs and nbs-of-nbs */
      for (ji = A_i[i] + 1; ji < A_i[i + 1]; ji++)
      {
         jj = A_j[ji];
         /* if not "decided" C or F */
         if (ma[jj] > -1)
         {
            /* if a candidate, remove jj from graph */
            if (ma[jj] > 0)
            {
               hypre_GraphRemove(list, head, tail, jj);
            }

            /* make jj an F point and mark "decided" */
            cf[jj] = fpt;
            ma[jj] = -1;

            for (jl = A_i[jj] + 1; jl < A_i[jj + 1]; jl++)
            {
               index = A_j[jl];
               /* if a candidate, increase ma */
               if (ma[index] > 0)
               {
                  ma[index]++;

                  /* move index in graph */
                  hypre_GraphRemove(list, head, tail, index);
                  hypre_GraphAdd(list, head, tail, index,
                                 (HYPRE_Int) ma[index]);
                  if (ma[index] > istack)
                  {
                     istack = (HYPRE_Int) ma[index];
                  }
               }
            }
         }
      }
      /* reset istack to point to biggest non-empty stack */
      for ( ; istack > 0; istack--)
      {
         /* if non-negative, break */
         if (head[-istack] > -1)
         {
            break;
         }
      }
   }

   hypre_TFree(ma, HYPRE_MEMORY_HOST);
   hypre_TFree(list, HYPRE_MEMORY_HOST);
   hypre_TFree(head_mem, HYPRE_MEMORY_HOST);
   hypre_TFree(tail_mem, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int hypre_IndepSetGreedyS(HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int n, HYPRE_Int *cf)
{
   Link *list;
   HYPRE_Int  *head, *head_mem, *ma;
   HYPRE_Int  *tail, *tail_mem;

   HYPRE_Int i, ji, jj, jl, index, istack, stack_size;

   ma = hypre_CTAlloc(HYPRE_Int,  n, HYPRE_MEMORY_HOST);

   /* Initialize the graph and measure array
    *
    * ma: cands >= 1
    *     cpts  = -1
    *     else  =  0
    * Note: only cands are put into graph */

   istack = 0;
   for (i = 0; i < n; i++)
   {
      if (cf[i] == cand)
      {
         ma[i] = 1;
         for (ji = A_i[i]; ji < A_i[i + 1]; ji++)
         {
            jj = A_j[ji];
            if (cf[jj] != cpt)
            {
               ma[i]++;
            }
         }
         if (ma[i] > istack)
         {
            istack = (HYPRE_Int) ma[i];
         }
      }
      else if (cf[i] == cpt)
      {
         ma[i] = -1;
      }
      else
      {
         ma[i] = 0;
      }
   }
   stack_size = 2 * istack;

   /* initialize graph */
   head_mem = hypre_CTAlloc(HYPRE_Int,  stack_size, HYPRE_MEMORY_HOST); head = head_mem + stack_size;
   tail_mem = hypre_CTAlloc(HYPRE_Int,  stack_size, HYPRE_MEMORY_HOST); tail = tail_mem + stack_size;
   list = hypre_CTAlloc(Link,  n, HYPRE_MEMORY_HOST);

   for (i = -1; i >= -stack_size; i--)
   {
      head[i] = i;
      tail[i] = i;
   }
   for (i = 0; i < n; i++)
   {
      if (ma[i] > 0)
      {
         hypre_GraphAdd(list, head, tail, i, (HYPRE_Int) ma[i]);
      }
   }

   /* Loop until all points are either F or C */
   while (istack > 0)
   {
      /* i w/ max measure at head of stacks */
      i = head[-istack];

      /* make i C point */
      cf[i] = cpt;
      ma[i] = -1;

      /* remove i from graph */
      hypre_GraphRemove(list, head, tail, i);

      /* update nbs and nbs-of-nbs */
      for (ji = A_i[i]; ji < A_i[i + 1]; ji++)
      {
         jj = A_j[ji];
         /* if not "decided" C or F */
         if (ma[jj] > -1)
         {
            /* if a candidate, remove jj from graph */
            if (ma[jj] > 0)
            {
               hypre_GraphRemove(list, head, tail, jj);
            }

            /* make jj an F point and mark "decided" */
            cf[jj] = fpt;
            ma[jj] = -1;

            for (jl = A_i[jj]; jl < A_i[jj + 1]; jl++)
            {
               index = A_j[jl];
               /* if a candidate, increase ma */
               if (ma[index] > 0)
               {
                  ma[index]++;

                  /* move index in graph */
                  hypre_GraphRemove(list, head, tail, index);
                  hypre_GraphAdd(list, head, tail, index,
                                 (HYPRE_Int) ma[index]);
                  if (ma[index] > istack)
                  {
                     istack = (HYPRE_Int) ma[index];
                  }
               }
            }
         }
      }
      /* reset istack to point to biggest non-empty stack */
      for ( ; istack > 0; istack--)
      {
         /* if non-negative, break */
         if (head[-istack] > -1)
         {
            break;
         }
      }
   }

   hypre_TFree(ma, HYPRE_MEMORY_HOST);
   hypre_TFree(list, HYPRE_MEMORY_HOST);
   hypre_TFree(head_mem, HYPRE_MEMORY_HOST);
   hypre_TFree(tail_mem, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* f point jac cr */
HYPRE_Int hypre_fptjaccr(HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data,
                         HYPRE_Int n, HYPRE_Real *e0, HYPRE_Real omega, HYPRE_Real *e1)
{
   HYPRE_Int i, j;
   HYPRE_Real res;

   for (i = 0; i < n; i++)
      if (cf[i] == fpt)
      {
         e0[i] = e1[i];
      }

   for (i = 0; i < n; i++)
   {
      res = 0.0e0;
      if (cf[i] == fpt)
      {
         for (j = A_i[i] + 1; j < A_i[i + 1]; j++)
         {
            if (cf[A_j[j]] == fpt)
            {
               res -= (A_data[j] * e0[A_j[j]]);
            }
         }
         e1[i] *= (1.0 - omega);
         e1[i] += omega * res / A_data[A_i[i]];
      }
   }
   return hypre_error_flag;
}


/* f point GS cr */
HYPRE_Int hypre_fptgscr(HYPRE_Int *cf, HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data,
                        HYPRE_Int n,
                        HYPRE_Real *e0, HYPRE_Real *e1)
{
   HYPRE_Int i, j;
   HYPRE_Real res;

   for (i = 0; i < n; i++)
      if (cf[i] == fpt)
      {
         e0[i] = e1[i];
      }

   for (i = 0; i < n; i++)
   {
      if (cf[i] == fpt)
      {
         res = 0.0e0;
         for ( j = A_i[i] + 1; j < A_i[i + 1]; j++)
         {
            if (cf[A_j[j]] == fpt)
            {
               res -= (A_data[j] * e1[A_j[j]]);
            }
         }
         e1[i] = res / A_data[A_i[i]];
      }
   }
   return hypre_error_flag;
}

/* form the candidate set U */
HYPRE_Int hypre_formu(HYPRE_Int *cf, HYPRE_Int n, HYPRE_Real *e1, HYPRE_Int *A_i, HYPRE_Real rho)
{
   HYPRE_Int i;
   HYPRE_Real candmeas = 0.0e0, max = 0.0e0;
   HYPRE_Real thresh = 1 - rho;

   for (i = 0; i < n; i++)
      if (hypre_abs(e1[i]) > max)
      {
         max = hypre_abs(e1[i]);
      }

   for (i = 0; i < n; i++)
   {
      if (cf[i] == fpt)
      {
         candmeas = hypre_pow(hypre_abs(e1[i]), 1.0) / max;
         if (candmeas > thresh && A_i[i + 1] - A_i[i] > 1)
         {
            cf[i] = cand;
         }
      }
   }
   return hypre_error_flag;
}
/*==========================================================================
 * Ruge's coarsening algorithm
 *==========================================================================*/

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define UNDECIDED 0


/**************************************************************
 *
 *      Ruge Coarsening routine
 *
 **************************************************************/
HYPRE_Int
hypre_BoomerAMGIndepRS( hypre_ParCSRMatrix    *S,
                        HYPRE_Int              measure_type,
                        HYPRE_Int              debug_flag,
                        HYPRE_Int             *CF_marker)
{
   MPI_Comm             comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_CSRMatrix     *S_diag        = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix     *S_offd        = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int           *S_i           = hypre_CSRMatrixI(S_diag);
   HYPRE_Int           *S_j           = hypre_CSRMatrixJ(S_diag);
   HYPRE_Int           *S_offd_i      = hypre_CSRMatrixI(S_offd);
   HYPRE_Int           *S_offd_j      = NULL;
   HYPRE_Int            num_variables = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int            num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   hypre_ParCSRCommHandle *comm_handle;
   hypre_CSRMatrix *ST;
   HYPRE_Int       *ST_i;
   HYPRE_Int       *ST_j;

   HYPRE_Int       *measure_array;
   HYPRE_Int       *CF_marker_offd;
   HYPRE_Int       *int_buf_data;

   HYPRE_Int        i, j, k, jS;
   HYPRE_Int        index;
   HYPRE_Int        num_procs, my_id;
   HYPRE_Int        num_sends = 0;
   HYPRE_Int        start, jrow;

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   HYPRE_Int       *lists, *where;
   HYPRE_Int        measure, new_meas;
   HYPRE_Int        num_left = 0;
   HYPRE_Int        nabor, nabor_two;

   HYPRE_Int        f_pnt = F_PT;
   HYPRE_Real       wall_time;

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
   where = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(S);

      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_cols_offd) { S_offd_j = hypre_CSRMatrixJ(S_offd); }

   jS = S_i[num_variables];

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, jS);
   ST_i = hypre_CTAlloc(HYPRE_Int, num_variables + 1, HYPRE_MEMORY_HOST);
   ST_j = hypre_CTAlloc(HYPRE_Int, jS, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixI(ST) = ST_i;
   hypre_CSRMatrixJ(ST) = ST_j;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i = 0; i <= num_variables; i++)
   {
      ST_i[i] = 0;
   }

   for (i = 0; i < jS; i++)
   {
      ST_i[S_j[i] + 1]++;
   }
   for (i = 0; i < num_variables; i++)
   {
      ST_i[i + 1] += ST_i[i];
   }
   for (i = 0; i < num_variables; i++)
   {
      for (j = S_i[i]; j < S_i[i + 1]; j++)
      {
         index = S_j[j];
         ST_j[ST_i[index]] = i;
         ST_i[index]++;
      }
   }
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i - 1];
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are given by the row sums of ST.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    * correct actual measures through adding influences from
    * neighbor processors
    *----------------------------------------------------------*/

   if (measure_type == 0)
   {
      measure_array = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_variables; i++)
      {
         measure_array[i] = 0;
      }
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] < 1)
         {
            for (j = S_i[i]; j < S_i[i + 1]; j++)
            {
               if (CF_marker[S_j[j]] < 1)
               {
                  measure_array[S_j[j]]++;
               }
            }
         }
      }

   }
   else
   {

      /* now the off-diagonal part of CF_marker */
      if (num_cols_offd)
      {
         CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      }
      else
      {
         CF_marker_offd = NULL;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         CF_marker_offd[i] = 0;
      }

      /*------------------------------------------------
       * Communicate the CF_marker values to the external nodes
       *------------------------------------------------*/
      int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                              num_sends), HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            int_buf_data[index++] = CF_marker[jrow];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      measure_array = hypre_CTAlloc(HYPRE_Int,  num_variables + num_cols_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_variables + num_cols_offd; i++)
      {
         measure_array[i] = 0;
      }

      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] < 1)
         {
            for (j = S_i[i]; j < S_i[i + 1]; j++)
            {
               if (CF_marker[S_j[j]] < 1)
               {
                  measure_array[S_j[j]]++;
               }
            }
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               if (CF_marker_offd[S_offd_j[j]] < 1)
               {
                  measure_array[num_variables + S_offd_j[j]]++;
               }
            }
         }
      }
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      /* now send those locally calculated values for the external nodes to the neighboring processors */
      if (num_procs > 1)
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                                                    &measure_array[num_variables], int_buf_data);

      /* finish the communication */
      if (num_procs > 1)
      {
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      /* now add the externally calculated part of the local nodes to the local nodes */
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)]
            += int_buf_data[index++];
      }
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }


   if (measure_type == 2 && num_procs > 1)
   {
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0)
         {
            if ((S_offd_i[i + 1] - S_offd_i[i]) == 0)
            {
               num_left++;
            }
            else
            {
               measure_array[i] = 0;
               CF_marker[i] = 2;
            }
         }
         else if (CF_marker[i] < 0)
         {
            measure_array[i] = 0;
         }
         else
         {
            measure_array[i] = -1;
         }
      }
   }
   else
   {
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0)
         {
            num_left++;
         }
         else if (CF_marker[i] < 0)
         {
            measure_array[i] = 0;
         }
         else
         {
            measure_array[i] = -1;
         }
      }
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   /* first coarsening phase */

   /*************************************************************
    *
    *   Initialize the lists
    *
    *************************************************************/

   for (j = 0; j < num_variables; j++)
   {
      measure = measure_array[j];
      if (CF_marker[j] == 0)
      {
         if (measure > 0)
         {
            hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
         }
         else
         {
            if (measure < 0) { hypre_printf("negative measure!\n"); }
            CF_marker[j] = f_pnt;
            for (k = S_i[j]; k < S_i[j + 1]; k++)
            {
               nabor = S_j[k];
               if (CF_marker[nabor] != SF_PT && CF_marker[nabor] < 1)
               {
                  if (nabor < j)
                  {
                     new_meas = measure_array[nabor];
                     if (new_meas > 0)
                        hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                           nabor, lists, where);

                     new_meas = ++(measure_array[nabor]);
                     hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                          nabor, lists, where);
                  }
                  else
                  {
                     new_meas = ++(measure_array[nabor]);
                  }
               }
            }
            --num_left;
         }
      }
   }

   /****************************************************************
    *
    *  Main loop of Ruge-Stueben first coloring pass.
    *
    *  WHILE there are still points to classify DO:
    *        1) find first point, i,  on list with max_measure
    *           make i a C-point, remove it from the lists
    *        2) For each point, j,  in S_i^T,
    *           a) Set j to be an F-point
    *           b) For each point, k, in S_j
    *                  move k to the list in LoL with measure one
    *                  greater than it occupies (creating new LoL
    *                  entry if necessary)
    *        3) For each point, j,  in S_i,
    *                  move j to the list in LoL with measure one
    *                  smaller than it occupies (creating new LoL
    *                  entry if necessary)
    *
    ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;

      CF_marker[index] = C_PT;
      measure = measure_array[index];
      measure_array[index] = 0;
      --num_left;

      hypre_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);

      for (j = ST_i[index]; j < ST_i[index + 1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = F_PT;
            measure = measure_array[nabor];

            hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_i[nabor] + 1; k < S_i[nabor + 1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  hypre_remove_point(&LoL_head, &LoL_tail, measure,
                                     nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);

                  hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                       nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_i[index]; j < S_i[index + 1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;

            if (measure > 0)
               hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor,
                                    lists, where);
            else
            {
               CF_marker[nabor] = F_PT;
               --num_left;

               for (k = S_i[nabor] + 1; k < S_i[nabor + 1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                        nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                          nabor_two, lists, where);
                  }
               }
            }
         }
      }
   }

   hypre_TFree(measure_array, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Coarsen 1st pass = %f\n",
                   my_id, wall_time);
   }

   if (measure_type == 2)
   {
      for (i = 0; i < num_variables; i++)
         if (CF_marker[i] == 2) { CF_marker[i] = 0; }
   }

   hypre_TFree(lists, HYPRE_MEMORY_HOST);
   hypre_TFree(where, HYPRE_MEMORY_HOST);
   hypre_TFree(LoL_head, HYPRE_MEMORY_HOST);
   hypre_TFree(LoL_tail, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/**************************************************************
 *
 *      Ruge Coarsening routine
 *
 **************************************************************/
HYPRE_Int
hypre_BoomerAMGIndepRSa( hypre_ParCSRMatrix    *S,
                         HYPRE_Int                    measure_type,
                         HYPRE_Int                    debug_flag,
                         HYPRE_Int                   *CF_marker)
{
   MPI_Comm             comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_CSRMatrix     *S_diag        = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix     *S_offd        = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int           *S_i           = hypre_CSRMatrixI(S_diag);
   HYPRE_Int           *S_j           = hypre_CSRMatrixJ(S_diag);
   HYPRE_Int           *S_offd_i      = hypre_CSRMatrixI(S_offd);
   HYPRE_Int           *S_offd_j      = NULL;
   HYPRE_Int            num_variables = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int            num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   hypre_ParCSRCommHandle *comm_handle;
   hypre_CSRMatrix *ST;
   HYPRE_Int       *ST_i;
   HYPRE_Int       *ST_j;

   HYPRE_Int       *measure_array;
   HYPRE_Int       *CF_marker_offd;
   HYPRE_Int       *int_buf_data;

   HYPRE_Int        i, j, k, jS;
   HYPRE_Int        index;
   HYPRE_Int        num_procs, my_id;
   HYPRE_Int        num_sends = 0;
   HYPRE_Int        start, jrow;

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   HYPRE_Int       *lists, *where;
   HYPRE_Int        measure, new_meas;
   HYPRE_Int        num_left = 0;
   HYPRE_Int        nabor, nabor_two;

   HYPRE_Int        f_pnt = F_PT;
   HYPRE_Real       wall_time;

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
   where = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(S);

      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_cols_offd) { S_offd_j = hypre_CSRMatrixJ(S_offd); }

   jS = S_i[num_variables];

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, jS);
   ST_i = hypre_CTAlloc(HYPRE_Int, num_variables + 1, HYPRE_MEMORY_HOST);
   ST_j = hypre_CTAlloc(HYPRE_Int, jS, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixI(ST) = ST_i;
   hypre_CSRMatrixJ(ST) = ST_j;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i = 0; i <= num_variables; i++)
   {
      ST_i[i] = 0;
   }

   for (i = 0; i < jS; i++)
   {
      ST_i[S_j[i] + 1]++;
   }
   for (i = 0; i < num_variables; i++)
   {
      ST_i[i + 1] += ST_i[i];
   }
   for (i = 0; i < num_variables; i++)
   {
      for (j = S_i[i]; j < S_i[i + 1]; j++)
      {
         index = S_j[j];
         ST_j[ST_i[index]] = i;
         ST_i[index]++;
      }
   }
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i - 1];
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
       * Compute the measures
       *
       * The measures are given by the row sums of ST.
       * Hence, measure_array[i] is the number of influences
       * of variable i.
       * correct actual measures through adding influences from
       * neighbor processors
       *----------------------------------------------------------*/

   if (measure_type == 0)
   {
      measure_array = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_variables; i++)
      {
         measure_array[i] = 0;
      }
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] < 1)
         {
            for (j = S_i[i] + 1; j < S_i[i + 1]; j++)
            {
               if (CF_marker[S_j[j]] < 1)
               {
                  measure_array[S_j[j]]++;
               }
            }
         }
      }

   }
   else
   {

      /* now the off-diagonal part of CF_marker */
      if (num_cols_offd)
      {
         CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      }
      else
      {
         CF_marker_offd = NULL;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         CF_marker_offd[i] = 0;
      }

      /*------------------------------------------------
       * Communicate the CF_marker values to the external nodes
       *------------------------------------------------*/
      int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                               num_sends), HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            int_buf_data[index++] = CF_marker[jrow];
         }
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      measure_array = hypre_CTAlloc(HYPRE_Int,  num_variables + num_cols_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_variables + num_cols_offd; i++)
      {
         measure_array[i] = 0;
      }

      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] < 1)
         {
            for (j = S_i[i] + 1; j < S_i[i + 1]; j++)
            {
               if (CF_marker[S_j[j]] < 1)
               {
                  measure_array[S_j[j]]++;
               }
            }
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               if (CF_marker_offd[S_offd_j[j]] < 1)
               {
                  measure_array[num_variables + S_offd_j[j]]++;
               }
            }
         }
      }
      hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
      /* now send those locally calculated values for the external nodes to the neighboring processors */
      if (num_procs > 1)
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                                                    &measure_array[num_variables], int_buf_data);

      /* finish the communication */
      if (num_procs > 1)
      {
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      /* now add the externally calculated part of the local nodes to the local nodes */
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)]
            += int_buf_data[index++];
      }
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }


   if (measure_type == 2 && num_procs > 1)
   {
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0)
         {
            if ((S_offd_i[i + 1] - S_offd_i[i]) == 0)
            {
               num_left++;
            }
            else
            {
               measure_array[i] = 0;
               CF_marker[i] = 2;
            }
         }
         else if (CF_marker[i] < 0)
         {
            measure_array[i] = 0;
         }
         else
         {
            measure_array[i] = -1;
         }
      }
   }
   else
   {
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0)
         {
            num_left++;
         }
         else if (CF_marker[i] < 0)
         {
            measure_array[i] = 0;
         }
         else
         {
            measure_array[i] = -1;
         }
      }
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }

   /* first coarsening phase */

   /*************************************************************
    *
    *   Initialize the lists
    *
    *************************************************************/

   for (j = 0; j < num_variables; j++)
   {
      measure = measure_array[j];
      if (CF_marker[j] == 0)
      {
         if (measure > 0)
         {
            hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
         }
         else
         {
            if (measure < 0) { hypre_printf("negative measure!\n"); }
            CF_marker[j] = f_pnt;
            for (k = S_i[j] + 1; k < S_i[j + 1]; k++)
            {
               nabor = S_j[k];
               if (CF_marker[nabor] != SF_PT && CF_marker[nabor] < 1)
               {
                  if (nabor < j)
                  {
                     new_meas = measure_array[nabor];
                     if (new_meas > 0)
                        hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                           nabor, lists, where);

                     new_meas = ++(measure_array[nabor]);
                     hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                          nabor, lists, where);
                  }
                  else
                  {
                     new_meas = ++(measure_array[nabor]);
                  }
               }
            }
            --num_left;
         }
      }
   }

   /****************************************************************
    *
    *  Main loop of Ruge-Stueben first coloring pass.
    *
    *  WHILE there are still points to classify DO:
    *        1) find first point, i,  on list with max_measure
    *           make i a C-point, remove it from the lists
    *        2) For each point, j,  in S_i^T,
    *           a) Set j to be an F-point
    *           b) For each point, k, in S_j
    *                  move k to the list in LoL with measure one
    *                  greater than it occupies (creating new LoL
    *                  entry if necessary)
    *        3) For each point, j,  in S_i,
    *                  move j to the list in LoL with measure one
    *                  smaller than it occupies (creating new LoL
    *                  entry if necessary)
    *
    ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;

      CF_marker[index] = C_PT;
      measure = measure_array[index];
      measure_array[index] = 0;
      --num_left;

      hypre_remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);

      for (j = ST_i[index] + 1; j < ST_i[index + 1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = F_PT;
            measure = measure_array[nabor];

            hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_i[nabor] + 1; k < S_i[nabor + 1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  hypre_remove_point(&LoL_head, &LoL_tail, measure,
                                     nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);

                  hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                       nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_i[index] + 1; j < S_i[index + 1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            hypre_remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;

            if (measure > 0)
               hypre_enter_on_lists(&LoL_head, &LoL_tail, measure, nabor,
                                    lists, where);
            else
            {
               CF_marker[nabor] = F_PT;
               --num_left;

               for (k = S_i[nabor] + 1; k < S_i[nabor + 1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     hypre_remove_point(&LoL_head, &LoL_tail, new_meas,
                                        nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);

                     hypre_enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                          nabor_two, lists, where);
                  }
               }
            }
         }
      }
   }

   hypre_TFree(measure_array, HYPRE_MEMORY_HOST);
   hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Coarsen 1st pass = %f\n",
                   my_id, wall_time);
   }

   if (measure_type == 2)
   {
      for (i = 0; i < num_variables; i++)
         if (CF_marker[i] == 2) { CF_marker[i] = 0; }
   }

   hypre_TFree(lists, HYPRE_MEMORY_HOST);
   hypre_TFree(where, HYPRE_MEMORY_HOST);
   hypre_TFree(LoL_head, HYPRE_MEMORY_HOST);
   hypre_TFree(LoL_tail, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}


HYPRE_Int
hypre_BoomerAMGIndepHMIS( hypre_ParCSRMatrix    *S,
                          HYPRE_Int              measure_type,
                          HYPRE_Int              debug_flag,
                          HYPRE_Int             *CF_marker)
{
   HYPRE_UNUSED_VAR(measure_type);

   HYPRE_Int    num_procs;
   MPI_Comm     comm = hypre_ParCSRMatrixComm(S);

   hypre_MPI_Comm_size(comm, &num_procs);

   /*-------------------------------------------------------
    * Perform Ruge coarsening followed by CLJP coarsening
    *-------------------------------------------------------*/

   hypre_BoomerAMGIndepRS(S, 2, debug_flag, CF_marker);

   if (num_procs > 1)
   {
      hypre_BoomerAMGIndepPMIS(S, 0, debug_flag, CF_marker);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGIndepHMISa( hypre_ParCSRMatrix    *S,
                           HYPRE_Int              measure_type,
                           HYPRE_Int              debug_flag,
                           HYPRE_Int             *CF_marker)
{
   HYPRE_UNUSED_VAR(measure_type);

   HYPRE_Int    num_procs;
   MPI_Comm     comm = hypre_ParCSRMatrixComm(S);

   hypre_MPI_Comm_size(comm, &num_procs);

   /*-------------------------------------------------------
    * Perform Ruge coarsening followed by CLJP coarsening
    *-------------------------------------------------------*/

   hypre_BoomerAMGIndepRSa(S, 2, debug_flag, CF_marker);

   if (num_procs > 1)
   {
      hypre_BoomerAMGIndepPMISa(S, 0, debug_flag, CF_marker);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------*/

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

/* begin HANS added */
/**************************************************************
 *
 *      Modified Independent Set Coarsening routine
 *          (don't worry about strong F-F connections
 *           without a common C point)
 *
 **************************************************************/
HYPRE_Int
hypre_BoomerAMGIndepPMIS( hypre_ParCSRMatrix    *S,
                          HYPRE_Int              CF_init,
                          HYPRE_Int              debug_flag,
                          HYPRE_Int             *CF_marker)
{
   MPI_Comm                comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg    *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix        *S_diag        = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int              *S_diag_i      = hypre_CSRMatrixI(S_diag);
   HYPRE_Int              *S_diag_j      = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix        *S_offd        = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int              *S_offd_i      = hypre_CSRMatrixI(S_offd);
   HYPRE_Int              *S_offd_j      = NULL;

   HYPRE_Int               num_variables = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int               num_cols_offd = 0;

   HYPRE_Int           num_sends = 0;
   HYPRE_Int          *int_buf_data;
   HYPRE_Real         *buf_data;

   HYPRE_Int          *CF_marker_offd;

   HYPRE_Real         *measure_array;
   HYPRE_Int          *graph_array;
   HYPRE_Int          *graph_array_offd;
   HYPRE_Int           graph_size;
   HYPRE_Int           graph_offd_size;
   HYPRE_BigInt        global_graph_size;

   HYPRE_Int           i, j, jj, jS, ig;
   HYPRE_Int           index, start, my_id, num_procs, jrow, cnt, elmt;


   HYPRE_Real       wall_time;



#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   /*******************************************************************************
    BEFORE THE INDEPENDENT SET COARSENING LOOP:
      measure_array: calculate the measures, and communicate them
        (this array contains measures for both local and external nodes)
      CF_marker, CF_marker_offd: initialize CF_marker
        (separate arrays for local and external; 0=unassigned, negative=F point, positive=C point)
   ******************************************************************************/

   /*--------------------------------------------------------------
    * Use the ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: S_data is not used; in stead, only strong columns are retained
    *       in S_j, which can then be used like S_data
    *----------------------------------------------------------------*/

   /*S_ext = NULL; */
   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(S);
      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                            num_sends), HYPRE_MEMORY_HOST);
   buf_data = hypre_CTAlloc(HYPRE_Real,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                         num_sends), HYPRE_MEMORY_HOST);

   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }

   /* now the off-diagonal part of CF_marker */
   if (num_cols_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      CF_marker_offd = NULL;
   }

   for (i = 0; i < num_cols_offd; i++)
   {
      CF_marker_offd[i] = 0;
   }

   /*------------------------------------------------
    * Communicate the CF_marker values to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         int_buf_data[index++] = CF_marker[jrow];
      }
   }

   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 CF_marker_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    * The measures are augmented by a random number
    * between 0 and 1.
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(HYPRE_Real,  num_variables + num_cols_offd, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_variables + num_cols_offd; i++)
   {
      measure_array[i] = 0;
   }

   /* calculate the local part for the local nodes */
   for (i = 0; i < num_variables; i++)
   {
      if (CF_marker[i] < 1)
      {
         for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
         {
            if (CF_marker[S_diag_j[j]] < 1)
            {
               measure_array[S_diag_j[j]] += 1.0;
            }
         }
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            if (CF_marker_offd[S_offd_j[j]] < 1)
            {
               measure_array[num_variables + S_offd_j[j]] += 1.0;
            }
         }
      }
   }

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   if (num_procs > 1)
      comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                                                 &measure_array[num_variables], buf_data);

   /* finish the communication */
   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)]
         += buf_data[index++];
   }

   /* set the measures of the external nodes to zero */
   for (i = num_variables; i < num_variables + num_cols_offd; i++)
   {
      measure_array[i] = 0;
   }

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   /* this augments the measures */
   i = 2747 + my_id;
   hypre_SeedRand(i);
   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] += hypre_Rand();
   }

   /*---------------------------------------------------
    * Initialize the graph arrays, and CF_marker arrays
    *---------------------------------------------------*/

   /* first the off-diagonal part of the graph array */
   if (num_cols_offd)
   {
      graph_array_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      graph_array_offd = NULL;
   }

   for (ig = 0; ig < num_cols_offd; ig++)
   {
      graph_array_offd[ig] = ig;
   }

   graph_offd_size = num_cols_offd;

   /* now the local part of the graph array, and the local CF_marker array */
   graph_array = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);

   if (CF_init == 1)
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         if ( (S_offd_i[i + 1] - S_offd_i[i]) > 0 || CF_marker[i] == -1)
         {
            CF_marker[i] = 0;
         }
         if (CF_marker[i] == SF_PT)
         {
            measure_array[i] = 0;
         }
         else if ( CF_marker[i] < 1)
         {
            if (measure_array[i] >= 1.0 )
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
               measure_array[i] = 0;
            }
         }
         else
         {
            measure_array[i] = 0;
         }
      }
   }
   else
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0)
         {
            if ( measure_array[i] >= 1.0 )
            {
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
            }
         }
         else
         {
            measure_array[i] = 0;
         }
      }
   }
   graph_size = cnt;

   /*------------------------------------------------
    * Communicate the local measures, which are complete,
    to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         buf_data[index++] = measure_array[jrow];
      }
   }

   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data,
                                                 &measure_array[num_variables]);

      hypre_ParCSRCommHandleDestroy(comm_handle);

   }

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Initialize CLJP phase = %f\n",
                   my_id, wall_time);
   }

   /*******************************************************************************
    THE INDEPENDENT SET COARSENING LOOP:
   ******************************************************************************/

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   while (1)
   {

      HYPRE_BigInt big_graph_size = (HYPRE_BigInt) graph_size;
      /* stop the coarsening if nothing left to be coarsened */
      hypre_MPI_Allreduce(&big_graph_size, &global_graph_size, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      if (global_graph_size == 0)
      {
         break;
      }

      /*     hypre_printf("\n");
             hypre_printf("*** MIS iteration %d\n",iter);
             hypre_printf("graph_size remaining %d\n",graph_size);*/

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       At the end, CF_marker is complete, but still needs to be
       communicated to CF_marker_offd
       *------------------------------------------------*/
      if (1)
      {
         /* hypre_BoomerAMGIndepSet(S, measure_array, graph_array,
            graph_size,
            graph_array_offd, graph_offd_size,
            CF_marker, CF_marker_offd);*/
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];
            if (measure_array[i] > 1)
            {
               CF_marker[i] = 1;
            }
         }
         for (ig = 0; ig < graph_offd_size; ig++)
         {
            i = graph_array_offd[ig];
            if (measure_array[i + num_variables] > 1)
            {
               CF_marker_offd[i] = 1;
            }
         }
         /*-------------------------------------------------------
          * Remove nodes from the initial independent set
          *-------------------------------------------------------*/

         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];
            if (measure_array[i] > 1)
            {
               for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
               {
                  j = S_diag_j[jS];

                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                     {
                        CF_marker[j] = 0;
                     }
                     else if (measure_array[j] > measure_array[i])
                     {
                        CF_marker[i] = 0;
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
               {
                  jj = S_offd_j[jS];
                  j = num_variables + jj;

                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                     {
                        CF_marker_offd[jj] = 0;
                     }
                     else if (measure_array[j] > measure_array[i])
                     {
                        CF_marker[i] = 0;
                     }
                  }
               }
            }
         }

         /*------------------------------------------------
          * Exchange boundary data for CF_marker: send internal
          points to external points
          *------------------------------------------------*/

         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                                                       CF_marker_offd, int_buf_data);

            hypre_ParCSRCommHandleDestroy(comm_handle);
         }

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
               if (!int_buf_data[index] && CF_marker[elmt] > 0)
               {
                  CF_marker[elmt] = 0;
                  index++;
               }
               else
               {
                  int_buf_data[index++] = CF_marker[elmt];
               }
            }
         }

         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                       CF_marker_offd);

            hypre_ParCSRCommHandleDestroy(comm_handle);
         }
      }

#if 0 /* debugging */
      iter++;
#endif
      /*------------------------------------------------
       * Set C-pts and F-pts.
       *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * If measure smaller than 1
          * make i an F point,
          *---------------------------------------------*/

         if (measure_array[i] < 1.)
         {
            /* set to be a F-pt */
            CF_marker[i] = F_PT;
         }

         /*---------------------------------------------
          * First treat the case where point i is in the
          * independent set: make i a C point,
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = C_PT;
         }

         /*---------------------------------------------
          * Now treat the case where point i is not in the
          * independent set: loop over
          * all the points j that influence equation i; if
          * j is a C point, then make i an F point.
          *---------------------------------------------*/

         else
         {

            /* first the local part */
            for (jS = S_diag_i[i]; jS < S_diag_i[i + 1]; jS++)
            {
               /* j is the column number, or the local number of the point influencing i */
               j = S_diag_j[jS];
               if (CF_marker[j] > 0)  /* j is a C-point */
               {
                  CF_marker[i] = F_PT;
               }
            }
            /* now the external part */
            for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
            {
               j = S_offd_j[jS];
               if (CF_marker_offd[j] > 0)  /* j is a C-point */
               {
                  CF_marker[i] = F_PT;
               }
            }

         } /* end else */
      } /* end first loop over graph */

      /* now communicate CF_marker to CF_marker_offd, to make
         sure that new external F points are known on this processor */

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
       points to external points
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      /*------------------------------------------------
       * Update subgraph
       *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         if (CF_marker[i] != 0) /* C or F point */
         {
            /* the independent set subroutine needs measure 0 for
               removed nodes */
            measure_array[i] = 0;
            /* take point out of the subgraph */
            graph_size--;
            graph_array[ig] = graph_array[graph_size];
            graph_array[graph_size] = i;
            ig--;
         }
      }
      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] != 0) /* C or F point */
         {
            /* the independent set subroutine needs measure 0 for
               removed nodes */
            measure_array[i + num_variables] = 0;
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }

   } /* end while */

   /*   hypre_printf("*** MIS iteration %d\n",iter);
        hypre_printf("graph_size remaining %d\n",graph_size);

        hypre_printf("num_cols_offd %d\n",num_cols_offd);
        for (i=0;i<num_variables;i++)
        {
        if(CF_marker[i]==1)
        hypre_printf("node %d CF %d\n",i,CF_marker[i]);
        }*/


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array, HYPRE_MEMORY_HOST);
   hypre_TFree(graph_array, HYPRE_MEMORY_HOST);
   if (num_cols_offd) { hypre_TFree(graph_array_offd, HYPRE_MEMORY_HOST); }
   hypre_TFree(buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGIndepPMISa( hypre_ParCSRMatrix    *S,
                           HYPRE_Int              CF_init,
                           HYPRE_Int              debug_flag,
                           HYPRE_Int             *CF_marker)
{
   MPI_Comm                comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg    *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle *comm_handle;

   hypre_CSRMatrix        *S_diag        = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int              *S_diag_i      = hypre_CSRMatrixI(S_diag);
   HYPRE_Int              *S_diag_j      = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix        *S_offd        = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int              *S_offd_i      = hypre_CSRMatrixI(S_offd);
   HYPRE_Int              *S_offd_j      = NULL;

   HYPRE_Int               num_variables = hypre_CSRMatrixNumRows(S_diag);
   HYPRE_Int               num_cols_offd = 0;

   HYPRE_Int           num_sends = 0;
   HYPRE_Int          *int_buf_data;
   HYPRE_Real         *buf_data;

   HYPRE_Int          *CF_marker_offd;

   HYPRE_Real         *measure_array;
   HYPRE_Int          *graph_array;
   HYPRE_Int          *graph_array_offd;
   HYPRE_Int           graph_size;
   HYPRE_Int           graph_offd_size;
   HYPRE_BigInt        global_graph_size;

   HYPRE_Int           i, j, jj, jS, ig;
   HYPRE_Int           index, start, my_id, num_procs, jrow, cnt, elmt;


   HYPRE_Real       wall_time;



#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   HYPRE_Int   iter = 0;
#endif

   /*******************************************************************************
    BEFORE THE INDEPENDENT SET COARSENING LOOP:
      measure_array: calculate the measures, and communicate them
        (this array contains measures for both local and external nodes)
      CF_marker, CF_marker_offd: initialize CF_marker
        (separate arrays for local and external; 0=unassigned, negative=F point, positive=C point)
   ******************************************************************************/

   /*--------------------------------------------------------------
    * Use the ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: S_data is not used; in stead, only strong columns are retained
    *       in S_j, which can then be used like S_data
    *----------------------------------------------------------------*/

   /*S_ext = NULL; */
   if (debug_flag == 3) { wall_time = time_getWallclockSeconds(); }
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (!comm_pkg)
   {
      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(S);

      comm_pkg = hypre_ParCSRMatrixCommPkg(S);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                            num_sends), HYPRE_MEMORY_HOST);
   buf_data = hypre_CTAlloc(HYPRE_Real,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                         num_sends), HYPRE_MEMORY_HOST);

   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }

   /* now the off-diagonal part of CF_marker */
   if (num_cols_offd)
   {
      CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      CF_marker_offd = NULL;
   }

   for (i = 0; i < num_cols_offd; i++)
   {
      CF_marker_offd[i] = 0;
   }

   /*------------------------------------------------
    * Communicate the CF_marker values to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         int_buf_data[index++] = CF_marker[jrow];
      }
   }

   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 CF_marker_offd);

      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    * The measures are augmented by a random number
    * between 0 and 1.
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(HYPRE_Real,  num_variables + num_cols_offd, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_variables + num_cols_offd; i++)
   {
      measure_array[i] = 0;
   }

   /* calculate the local part for the local nodes */
   for (i = 0; i < num_variables; i++)
   {
      if (CF_marker[i] < 1)
      {
         for (j = S_diag_i[i] + 1; j < S_diag_i[i + 1]; j++)
         {
            if (CF_marker[S_diag_j[j]] < 1)
            {
               measure_array[S_diag_j[j]] += 1.0;
            }
         }
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            if (CF_marker_offd[S_offd_j[j]] < 1)
            {
               measure_array[num_variables + S_offd_j[j]] += 1.0;
            }
         }
      }
   }

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   if (num_procs > 1)
      comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg,
                                                 &measure_array[num_variables], buf_data);

   /* finish the communication */
   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)]
         += buf_data[index++];
   }

   /* set the measures of the external nodes to zero */
   for (i = num_variables; i < num_variables + num_cols_offd; i++)
   {
      measure_array[i] = 0;
   }

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   /* this augments the measures */
   i = 2747 + my_id;
   hypre_SeedRand(i);
   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] += hypre_Rand();
   }

   /*---------------------------------------------------
    * Initialize the graph arrays, and CF_marker arrays
    *---------------------------------------------------*/

   /* first the off-diagonal part of the graph array */
   if (num_cols_offd)
   {
      graph_array_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   }
   else
   {
      graph_array_offd = NULL;
   }

   for (ig = 0; ig < num_cols_offd; ig++)
   {
      graph_array_offd[ig] = ig;
   }

   graph_offd_size = num_cols_offd;

   /* now the local part of the graph array, and the local CF_marker array */
   graph_array = hypre_CTAlloc(HYPRE_Int,  num_variables, HYPRE_MEMORY_HOST);

   if (CF_init == 1)
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         if ( (S_offd_i[i + 1] - S_offd_i[i]) > 0 || CF_marker[i] == -1)
         {
            CF_marker[i] = 0;
         }
         if (CF_marker[i] == SF_PT)
         {
            measure_array[i] = 0;
         }
         else if ( CF_marker[i] < 1)
         {
            if (measure_array[i] >= 1.0 )
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
               measure_array[i] = 0;
            }
         }
         else
         {
            measure_array[i] = 0;
         }
      }
   }
   else
   {
      cnt = 0;
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0 && measure_array[i] >= 1.0 )
         {
            graph_array[cnt++] = i;
         }
         else
         {
            measure_array[i] = 0;
         }
      }
   }
   graph_size = cnt;

   /*------------------------------------------------
    * Communicate the local measures, which are complete,
    to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         buf_data[index++] = measure_array[jrow];
      }
   }

   if (num_procs > 1)
   {
      comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data,
                                                 &measure_array[num_variables]);

      hypre_ParCSRCommHandleDestroy(comm_handle);

   }

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d    Initialize CLJP phase = %f\n",
                   my_id, wall_time);
   }

   /*******************************************************************************
    THE INDEPENDENT SET COARSENING LOOP:
   ******************************************************************************/

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   while (1)
   {

      HYPRE_BigInt big_graph_size = (HYPRE_BigInt) graph_size;
      /* stop the coarsening if nothing left to be coarsened */
      hypre_MPI_Allreduce(&big_graph_size, &global_graph_size, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      if (global_graph_size == 0)
      {
         break;
      }

      /*     hypre_printf("\n");
             hypre_printf("*** MIS iteration %d\n",iter);
             hypre_printf("graph_size remaining %d\n",graph_size);*/

      /*------------------------------------------------
       * Pick an independent set of points with
       * maximal measure.
       At the end, CF_marker is complete, but still needs to be
       communicated to CF_marker_offd
       *------------------------------------------------*/
      if (1)
      {
         /* hypre_BoomerAMGIndepSet(S, measure_array, graph_array,
            graph_size,
            graph_array_offd, graph_offd_size,
            CF_marker, CF_marker_offd);*/
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];
            if (measure_array[i] > 1)
            {
               CF_marker[i] = 1;
            }
         }
         for (ig = 0; ig < graph_offd_size; ig++)
         {
            i = graph_array_offd[ig];
            if (measure_array[i + num_variables] > 1)
            {
               CF_marker_offd[i] = 1;
            }
         }
         /*-------------------------------------------------------
          * Remove nodes from the initial independent set
          *-------------------------------------------------------*/

         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];
            if (measure_array[i] > 1)
            {
               for (jS = S_diag_i[i] + 1; jS < S_diag_i[i + 1]; jS++)
               {
                  j = S_diag_j[jS];

                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                     {
                        CF_marker[j] = 0;
                     }
                     else if (measure_array[j] > measure_array[i])
                     {
                        CF_marker[i] = 0;
                     }
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
               {
                  jj = S_offd_j[jS];
                  j = num_variables + jj;

                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                     {
                        CF_marker_offd[jj] = 0;
                     }
                     else if (measure_array[j] > measure_array[i])
                     {
                        CF_marker[i] = 0;
                     }
                  }
               }
            }
         }

         /*------------------------------------------------
          * Exchange boundary data for CF_marker: send internal
          points to external points
          *------------------------------------------------*/

         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                                                       CF_marker_offd, int_buf_data);

            hypre_ParCSRCommHandleDestroy(comm_handle);
         }

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
               if (!int_buf_data[index] && CF_marker[elmt] > 0)
               {
                  CF_marker[elmt] = 0;
                  index++;
               }
               else
               {
                  int_buf_data[index++] = CF_marker[elmt];
               }
            }
         }

         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                       CF_marker_offd);

            hypre_ParCSRCommHandleDestroy(comm_handle);
         }
      }

#if 0 /* debugging */
      iter++;
#endif
      /*------------------------------------------------
       * Set C-pts and F-pts.
       *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         /*---------------------------------------------
          * First treat the case where point i is in the
          * independent set: make i a C point,
          *---------------------------------------------*/

         if (CF_marker[i] > 0)
         {
            /* set to be a C-pt */
            CF_marker[i] = C_PT;
         }

         /*---------------------------------------------
          * Now treat the case where point i is not in the
          * independent set: loop over
          * all the points j that influence equation i; if
          * j is a C point, then make i an F point.
          *---------------------------------------------*/

         else
         {

            /* first the local part */
            for (jS = S_diag_i[i] + 1; jS < S_diag_i[i + 1]; jS++)
            {
               /* j is the column number, or the local number of the point influencing i */
               j = S_diag_j[jS];
               if (CF_marker[j] > 0)  /* j is a C-point */
               {
                  CF_marker[i] = F_PT;
               }
            }
            /* now the external part */
            for (jS = S_offd_i[i]; jS < S_offd_i[i + 1]; jS++)
            {
               j = S_offd_j[jS];
               if (CF_marker_offd[j] > 0)  /* j is a C-point */
               {
                  CF_marker[i] = F_PT;
               }
            }

         } /* end else */
      } /* end first loop over graph */

      /* now communicate CF_marker to CF_marker_offd, to make
         sure that new external F points are known on this processor */

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
       points to external points
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    CF_marker_offd);

         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      /*------------------------------------------------
       * Update subgraph
       *------------------------------------------------*/

      for (ig = 0; ig < graph_size; ig++)
      {
         i = graph_array[ig];

         if (CF_marker[i] != 0) /* C or F point */
         {
            /* the independent set subroutine needs measure 0 for
               removed nodes */
            measure_array[i] = 0;
            /* take point out of the subgraph */
            graph_size--;
            graph_array[ig] = graph_array[graph_size];
            graph_array[graph_size] = i;
            ig--;
         }
      }
      for (ig = 0; ig < graph_offd_size; ig++)
      {
         i = graph_array_offd[ig];

         if (CF_marker_offd[i] != 0) /* C or F point */
         {
            /* the independent set subroutine needs measure 0 for
               removed nodes */
            measure_array[i + num_variables] = 0;
            /* take point out of the subgraph */
            graph_offd_size--;
            graph_array_offd[ig] = graph_array_offd[graph_offd_size];
            graph_array_offd[graph_offd_size] = i;
            ig--;
         }
      }

   } /* end while */

   /*   hypre_printf("*** MIS iteration %d\n",iter);
        hypre_printf("graph_size remaining %d\n",graph_size);

        hypre_printf("num_cols_offd %d\n",num_cols_offd);
        for (i=0;i<num_variables;i++)
        {
        if(CF_marker[i]==1)
        hypre_printf("node %d CF %d\n",i,CF_marker[i]);
        }*/


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array, HYPRE_MEMORY_HOST);
   hypre_TFree(graph_array, HYPRE_MEMORY_HOST);
   if (num_cols_offd) { hypre_TFree(graph_array_offd, HYPRE_MEMORY_HOST); }
   hypre_TFree(buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
HYPRE_Int
hypre_BoomerAMGCoarsenCR( hypre_ParCSRMatrix    *A,
                          hypre_IntArray   **CF_marker_ptr,
                          HYPRE_BigInt      *coarse_size_ptr,
                          HYPRE_Int          num_CR_relax_steps,
                          HYPRE_Int          IS_type,
                          HYPRE_Int          num_functions,
                          HYPRE_Int          rlx_type,
                          HYPRE_Real         relax_weight,
                          HYPRE_Real         omega,
                          HYPRE_Real         theta,
                          HYPRE_Solver       smoother,
                          hypre_ParCSRMatrix *AN,
                          HYPRE_Int          useCG,
                          hypre_ParCSRMatrix *S)
/*HYPRE_Int                CRaddCpoints)*/
{
   /* HYPRE_Real theta_global;*/
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix *S_diag = hypre_ParCSRMatrixDiag(S);
   HYPRE_BigInt     global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt    *row_starts = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int       *A_i           = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_j           = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int       *S_i           = hypre_CSRMatrixI(S_diag);
   HYPRE_Int       *S_j           = hypre_CSRMatrixJ(S_diag);
   /*HYPRE_Real      *A_data        = hypre_CSRMatrixData(A_diag);*/
   /*HYPRE_Real      *Vtemp_data        = hypre_CSRMatrixData(A_diag);*/
   HYPRE_Real      *Vtemp_data;
   HYPRE_Real      *Ptemp_data;
   HYPRE_Real      *Ztemp_data;
   HYPRE_Int        num_variables = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int       *A_offd_i     = hypre_CSRMatrixI(A_offd);
   hypre_ParVector *e0_vec, *e1_vec, *Vtemp, *Ptemp;

   hypre_ParVector *e2_vec;
   hypre_ParVector *Rtemp, *Qtemp, *Ztemp;
   HYPRE_Int       *AN_i, *AN_offd_i;
   HYPRE_Int       *CF_marker;
   /*HYPRE_Int             *CFN_marker;*/
   HYPRE_BigInt     coarse_size;
   HYPRE_Int        i, j, jj, j2, nstages = 0;
   HYPRE_Int        num_procs, my_id, num_threads;
   HYPRE_Int        num_nodes = num_variables / num_functions;
   HYPRE_Real       rho = 1.0;
   HYPRE_Real       gamma = 0.0;
   HYPRE_Real       rho0, rho1, *e0, *e1, *sum = NULL;
   HYPRE_Real       rho_old, relrho;
   HYPRE_Real       *e2;
   HYPRE_Real       alpha, beta, gammaold;
   HYPRE_Int        num_coarse;
   HYPRE_BigInt     global_num_variables, global_nc = 0;
   HYPRE_Real candmeas = 0.0e0, local_max = 0.0e0, global_max = 0;
   /*HYPRE_Real thresh=1-rho;*/
   HYPRE_Real thresh = 0.5;

   hypre_ParVector    *Relax_temp = NULL;



   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   num_threads = hypre_NumThreads();


   global_num_variables = hypre_ParCSRMatrixGlobalNumRows(A);
   /*if(CRaddCpoints == 0)
     {*/
   if (num_functions == 1)
   {
      *CF_marker_ptr = hypre_IntArrayCreate(num_variables);
   }
   else
   {
      num_nodes = num_variables / num_functions;
      sum = hypre_CTAlloc(HYPRE_Real,  num_nodes, HYPRE_MEMORY_HOST);
      *CF_marker_ptr = hypre_IntArrayCreate(num_nodes);
   }
   hypre_IntArrayInitialize(*CF_marker_ptr);
   hypre_IntArraySetConstantValues(*CF_marker_ptr, fpt);
   CF_marker = hypre_IntArrayData(*CF_marker_ptr);
   /*}
     else
     {
     CF_marker = *CF_marker_ptr;*/
   /*CF_marker = hypre_CTAlloc(HYPRE_Int, num_variables);
     for ( i = 0; i < num_variables; i++)
     CF_marker[i] = fpt;
     num_nodes = num_variables/num_functions;
     CFN_marker = hypre_CTAlloc(HYPRE_Int,  num_nodes, HYPRE_MEMORY_HOST);
     sum = hypre_CTAlloc(HYPRE_Real,  num_nodes, HYPRE_MEMORY_HOST);
     for ( i = 0; i < num_nodes; i++)
     CFN_marker[i] = fpt;*/
   /*}*/

   /* Run the CR routine */

   if (my_id == 0) { hypre_fprintf(stdout, "\n... Building CF using CR ...\n\n"); }
   /*cr(A_i, A_j, A_data, num_variables, CF_marker,
     RelaxScheme1, omega1, theta_global1,mu1);*/

   /* main cr routine */
   /*HYPRE_Int cr(HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data, HYPRE_Int n, HYPRE_Int *cf,
     HYPRE_Int rlx, HYPRE_Real omega, HYPRE_Real tg, HYPRE_Int mu)*/

   e0_vec = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(e0_vec);
   e1_vec = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(e1_vec);
   e2_vec = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(e2_vec);
   Vtemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(Vtemp);
   Vtemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));
   Ptemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(Ptemp);
   Ptemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ptemp));
   Qtemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(Qtemp);
   Ztemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(Ztemp);
   Ztemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Ztemp));
   Rtemp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
   hypre_ParVectorInitialize(Rtemp);

   if (num_threads > 1)
   {
      Relax_temp = hypre_ParVectorCreate(comm, global_num_rows, row_starts);
      hypre_ParVectorInitialize(Relax_temp);
   }

   e0 = hypre_VectorData(hypre_ParVectorLocalVector(e0_vec));
   e1 = hypre_VectorData(hypre_ParVectorLocalVector(e1_vec));
   e2 = hypre_VectorData(hypre_ParVectorLocalVector(e2_vec));

   if (my_id == 0)
   {
      hypre_fprintf(stdout, "Stage  \t rho \t alpha \n");
      hypre_fprintf(stdout, "-----------------------\n");
   }

   for (i = 0; i < num_variables; i++)
   {
      e1[i] = 1.0e0;
   }
   /*e1[i] = 1.0e0+.1*hypre_RandI();*/

   /* stages */
   while (1)
   {
      if (nstages > 0)
      {
         if (num_functions == 1)
         {
            for (i = 0; i < num_variables; i++)
            {
               Vtemp_data[i] = 0.0e0;
               if (CF_marker[i] == cpt)
               {
                  e0[i] = 0.0e0;
                  e1[i] = 0.0e0;
               }
            }
         }
         else
         {
            jj = 0;
            for (i = 0; i < num_nodes; i++)
            {
               for (j = 0; j < num_functions; j++)
               {
                  if (CF_marker[i] == cpt)
                  {
                     e0[jj] = 0.0e0;
                     e1[jj] = 0.0e0;
                  }
                  Vtemp_data[jj++] = 0.0e0;
               }
            }
         }
      }

      /*for (i=0;i<num_CR_relax_steps;i++)
        fptgscr(CF_marker,A_i,A_j,A_data,num_variables,e0,e1); */
      /*switch(rlx_type){
        case fptOmegaJac:
        for (i=0;i<mu;i++)
        fptjaccr(cf,A_i,A_j,A_data,n,e0,omega,e1);
        break;
        case fptgs:
        for (i=0;i<mu;i++)
        fptgscr(cf,A_i,A_j,A_data,n,e0,e1);
        break;
        }*/

      if (smoother)
      {
         for (i = 0; i < num_CR_relax_steps; i++)
         {
            jj = 0;
            for (j = 0; j < num_nodes; j++)
            {
               for (j2 = 0; j2 < num_functions; j2++)
               {
                  if (CF_marker[j] == fpt) { e0[jj] = e1[jj]; }
                  jj++;
               }
            }
            hypre_SchwarzCFSolve((void *)smoother, A, Vtemp, e1_vec,
                                 CF_marker, fpt);
         }
      }
      else
      {
         rho = 1;
         rho_old = 1;
         relrho = 1.;
         i = 0;
         while (rho >= 0.1 * theta && (i < num_CR_relax_steps || relrho >= 0.1))
            /*for (i=0;i<num_CR_relax_steps;i++)*/
         {
            for (j = 0; j < num_variables; j++)
               if (CF_marker[j] == fpt) { e0[j] = e1[j]; }
            hypre_BoomerAMGRelax(A, Vtemp, CF_marker,
                                 rlx_type, fpt,
                                 relax_weight, omega, NULL,
                                 e1_vec, e0_vec,
                                 Relax_temp);
            /*if (i==num_CR_relax_steps-1) */
            if (i == 1)
            {
               for (j = 0; j < num_variables; j++)
                  if (CF_marker[j] == fpt) { e2[j] = e1[j]; }
            }
            rho0 = hypre_ParVectorInnerProd(e0_vec, e0_vec);
            rho1 = hypre_ParVectorInnerProd(e1_vec, e1_vec);
            rho_old = rho;
            rho = hypre_sqrt(rho1) / hypre_sqrt(rho0);
            relrho = hypre_abs(rho - rho_old) / rho;
            i++;
         }
      }
      /*rho=0.0e0; rho0=0.0e0; rho1=0.0e0;*/
      /*for(i=0;i<num_variables;i++){
        rho0 += hypre_pow(e0[i],2);
        rho1 += hypre_pow(e1[i],2);
        }*/

      /*rho0 = hypre_ParVectorInnerProd(e0_vec,e0_vec);
        rho1 = hypre_ParVectorInnerProd(e1_vec,e1_vec);
        rho = hypre_sqrt(rho1)/hypre_sqrt(rho0);*/
      for (j = 0; j < num_variables; j++)
         if (CF_marker[j] == fpt) { e1[j] = e2[j]; }
      if (rho > theta)
      {
         if (useCG)
         {
            for (i = 0; i < num_variables; i++)
            {
               if (CF_marker[i] ==  fpt)
               {
                  e1[i] = 1.0e0;
                  /*e1[i] = 1.0e0+.1*hypre_RandI();*/
                  e0[i] = e1[i];
               }
            }

            hypre_ParVectorSetConstantValues(Rtemp, 0);
            rho1 = hypre_ParVectorInnerProd(e1_vec, e1_vec);
            rho0 = rho1;
            i = 0;
            while (rho1 / rho0 > 1.e-2 && i < num_CR_relax_steps)
            {
               if (i == 0)
               {
                  hypre_ParCSRMatrixMatvec_FF(-1.0, A, e0_vec, 0.0, Rtemp, CF_marker, fpt);
               }
               /*hypre_BoomerAMGRelax(A, Rtemp, CF_marker, rlx_type, fpt,
                 relax_weight, omega, NULL, Ztemp, Vtemp);*/
               HYPRE_ParCSRDiagScale(NULL, (HYPRE_ParCSRMatrix) A, (HYPRE_ParVector) Rtemp,
                                     (HYPRE_ParVector) Ztemp);
               gammaold = gamma;
               gamma = hypre_ParVectorInnerProd(Rtemp, Ztemp);
               if (i == 0)
               {
                  hypre_ParVectorCopy(Ztemp, Ptemp);
                  beta = 1.0;
               }
               else
               {
                  beta = gamma / gammaold;
                  for (j = 0; j < num_variables; j++)
                     if (CF_marker[j] == fpt)
                     {
                        Ptemp_data[j] = Ztemp_data[j] + beta * Ptemp_data[j];
                     }
               }
               hypre_ParCSRMatrixMatvec_FF(1.0, A, Ptemp, 0.0, Qtemp, CF_marker, fpt);
               alpha = gamma / hypre_ParVectorInnerProd(Ptemp, Qtemp);
               hypre_ParVectorAxpy(-alpha, Qtemp, Rtemp);
               for (j = 0; j < num_variables; j++)
                  if (CF_marker[j] == fpt) { e0[j] = e1[j]; }
               hypre_ParVectorAxpy(-alpha, Ptemp, e1_vec);
               rho1 = hypre_ParVectorInnerProd(e1_vec, e1_vec);
               i++;
            }
         }
         /*formu(CF_marker,num_variables,e1,A_i,rho);*/
         if (nstages)
         {
            thresh = 0.5;
         }
         else
         {
            thresh = 0.3;
         }
         for (i = 1; i < num_CR_relax_steps; i++)
         {
            thresh *= 0.3;
         }
         /*thresh=0.1;*/

         if (num_functions == 1)
            /*if(CRaddCpoints == 0)*/
         {
            local_max = 0.0;
            for (i = 0; i < num_variables; i++)
               if (hypre_abs(e1[i]) > local_max)
               {
                  local_max = hypre_abs(e1[i]);
               }
         }
         else
         {
            jj = 0;
            local_max = 0.0;
            for (i = 0; i < num_nodes; i++)
            {
               /*CF_marker[jj] = CFN_marker[i];*/
               sum[i] = hypre_abs(e1[jj++]);
               for (j = 1; j < num_functions; j++)
               {
                  /*CF_marker[jj] = CFN_marker[i];*/
                  sum[i] += hypre_abs(e1[jj++]);
               }
               if (sum[i] > local_max)
               {
                  local_max = sum[i];
               }
            }
         }

         hypre_MPI_Allreduce(&local_max, &global_max, 1, HYPRE_MPI_REAL, hypre_MPI_MAX, comm);
         if (num_functions == 1)
            /*if(CRaddCpoints == 0)*/
         {
            for (i = 0; i < num_variables; i++)
            {
               if (CF_marker[i] == fpt)
               {
                  candmeas = hypre_pow(hypre_abs(e1[i]), 1.0) / global_max;
                  if (candmeas > thresh &&
                      (A_i[i + 1] - A_i[i] + A_offd_i[i + 1] - A_offd_i[i]) > 1)
                  {
                     CF_marker[i] = cand;
                  }
               }
            }
            if (IS_type == 1)
            {
               hypre_BoomerAMGIndepHMIS(S, 0, 0, CF_marker);
            }
            else if (IS_type == 7)
            {
               hypre_BoomerAMGIndepHMISa(A, 0, 0, CF_marker);
            }
            else if (IS_type == 2)
            {
               hypre_BoomerAMGIndepPMISa(A, 0, 0, CF_marker);
            }
            else if (IS_type == 5)
            {
               hypre_BoomerAMGIndepPMIS(S, 0, 0, CF_marker);
            }
            else if (IS_type == 3)
            {
               hypre_IndepSetGreedy(A_i, A_j, num_variables, CF_marker);
            }
            else if (IS_type == 6)
            {
               hypre_IndepSetGreedyS(S_i, S_j, num_variables, CF_marker);
            }
            else if (IS_type == 4)
            {
               hypre_BoomerAMGIndepRS(S, 1, 0, CF_marker);
            }
            else
            {
               hypre_BoomerAMGIndepRSa(A, 1, 0, CF_marker);
            }
         }
         else
         {
            AN_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AN));
            AN_offd_i     = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(AN));

            for (i = 0; i < num_nodes; i++)
            {
               /*if (CFN_marker[i] == fpt)*/
               if (CF_marker[i] == fpt)
               {
                  candmeas = sum[i] / global_max;
                  if (candmeas > thresh &&
                      (AN_i[i + 1] - AN_i[i] + AN_offd_i[i + 1] - AN_offd_i[i]) > 1)
                  {
                     /*CFN_marker[i] = cand; */
                     CF_marker[i] = cand;
                  }
               }
            }
            if (IS_type == 1)
            {
               hypre_BoomerAMGIndepHMIS(AN, 0, 0, CF_marker);
            }
            /*hypre_BoomerAMGIndepHMIS(AN,0,0,CFN_marker);*/
            else if (IS_type == 2)
            {
               hypre_BoomerAMGIndepPMIS(AN, 0, 0, CF_marker);
            }
            /*hypre_BoomerAMGIndepPMIS(AN,0,0,CFN_marker);*/
            else if (IS_type == 3)
            {
               hypre_IndepSetGreedy(hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(AN)),
                                    hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(AN)),
                                    num_nodes, CF_marker);
               /*num_nodes,CFN_marker);*/
            }
            else
            {
               hypre_BoomerAMGIndepRS(AN, 1, 0, CF_marker);
            }
            /*hypre_BoomerAMGIndepRS(AN,1,0,CFN_marker);*/
         }

         if (my_id == 0) hypre_fprintf(stdout, "  %d \t%2.3f  \t%2.3f \n",
                                          nstages, rho, (HYPRE_Real)global_nc / (HYPRE_Real)global_num_variables);
         /* update for next sweep */
         num_coarse = 0;
         if (num_functions == 1)
            /*if(CRaddCpoints == 0)*/
         {
            for (i = 0; i < num_variables; i++)
            {
               if (CF_marker[i] ==  cpt)
               {
                  num_coarse++;
               }
               else if (CF_marker[i] ==  fpt)
               {
                  e0[i] = 1.0e0 + .1 * hypre_RandI();
                  e1[i] = 1.0e0 + .1 * hypre_RandI();
               }
            }
         }
         else
         {
            jj = 0;
            for (i = 0; i < num_nodes; i++)
            {
               /*if (CFN_marker[i] ==  cpt) */
               if (CF_marker[i] ==  cpt)
               {
                  num_coarse++;
                  jj += num_functions;
                  /*for (j=0; j < num_functions; j++)
                    CF_marker[jj++] = CFN_marker[i];*/
               }
               /*else if (CFN_marker[i] ==  fpt)*/
               else if (CF_marker[i] ==  fpt)
               {
                  for (j = 0; j < num_functions; j++)
                  {
                     /*CF_marker[jj] = CFN_marker[i];
                       e0[jj] = 1.0e0+.1*hypre_RandI();
                       e1[jj++] = 1.0e0+.1*hypre_RandI();*/
                     e0[jj] = 1.0e0;
                     e1[jj++] = 1.0e0;
                  }
               }
               /*else
                 {
                 for (j=0; j < num_functions; j++)
                 CF_marker[jj++] = CFN_marker[i];
                 } */
            }
         }
         nstages += 1;
         hypre_MPI_Allreduce(&num_coarse, &global_nc, 1, HYPRE_MPI_INT, hypre_MPI_MAX, comm);
      }
      else
      {
         if (my_id == 0) hypre_fprintf(stdout, "  %d \t%2.3f  \t%2.3f \n",
                                          nstages, rho, (HYPRE_Real)global_nc / (HYPRE_Real)global_num_variables);
         break;
      }
   }
   hypre_ParVectorDestroy(e0_vec);
   hypre_ParVectorDestroy(e1_vec);
   hypre_ParVectorDestroy(e2_vec);
   hypre_ParVectorDestroy(Vtemp);
   hypre_ParVectorDestroy(Ptemp);
   hypre_ParVectorDestroy(Qtemp);
   hypre_ParVectorDestroy(Rtemp);
   hypre_ParVectorDestroy(Ztemp);

   if (num_threads > 1)
   {
      hypre_ParVectorDestroy(Relax_temp);
   }



   if (my_id == 0) { hypre_fprintf(stdout, "\n... Done \n\n"); }
   coarse_size = 0;
   for ( i = 0 ; i < num_variables; i++)
   {
      if ( CF_marker[i] == cpt)
      {
         coarse_size++;
      }
   }
   /*if(CRaddCpoints) hypre_TFree(CFN_marker);*/
   *coarse_size_ptr = coarse_size;
   hypre_TFree(sum, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}
