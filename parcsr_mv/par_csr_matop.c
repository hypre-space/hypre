/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/

#include "headers.h"


/* -----------------------------------------------------------------------------
 * generate a parallel spanning tree (for Maxwell Equation)
 * G_csr is the node to edge connectivity matrix
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixGenSpanningTree(hypre_ParCSRMatrix *G_csr, int **indices,
                                       int G_type)
{
   int nrows_G, ncols_G, *G_diag_i, *G_diag_j, *GT_diag_mat, i, j, k, edge;
   int *nodes_marked, *edges_marked, *queue, queue_tail, queue_head, node;
   int mypid, nprocs, n_children, *children, nsends, *send_procs, *recv_cnts;
   int nrecvs, *recv_procs, n_proc_array, *proc_array, *pgraph_i, *pgraph_j;
   int parent, proc, proc2, node2, found, *t_indices, tree_size, *T_diag_i;
   int *T_diag_j, *counts, offset;
   MPI_Comm            comm;
   hypre_ParCSRCommPkg *comm_pkg;
   hypre_CSRMatrix     *G_diag;

   /* fetch G matrix (G_type = 0 ==> node to edge) */

   if (G_type == 0)
   {
      nrows_G = hypre_ParCSRMatrixGlobalNumRows(G_csr);
      ncols_G = hypre_ParCSRMatrixGlobalNumCols(G_csr);
      G_diag = hypre_ParCSRMatrixDiag(G_csr);
      G_diag_i = hypre_CSRMatrixI(G_diag);
      G_diag_j = hypre_CSRMatrixJ(G_diag);
   }
   else
   {
      nrows_G = hypre_ParCSRMatrixGlobalNumCols(G_csr);
      ncols_G = hypre_ParCSRMatrixGlobalNumRows(G_csr);
      G_diag = hypre_ParCSRMatrixDiag(G_csr);
      T_diag_i = hypre_CSRMatrixI(G_diag);
      T_diag_j = hypre_CSRMatrixJ(G_diag);
      counts = (int *) malloc(nrows_G * sizeof(int));
      for (i = 0; i < nrows_G; i++) counts[i] = 0;
      for (i = 0; i < T_diag_i[ncols_G]; i++) counts[T_diag_j[i]]++;
      G_diag_i = (int *) malloc((nrows_G+1) * sizeof(int));
      G_diag_j = (int *) malloc(T_diag_i[ncols_G] * sizeof(int));
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++) G_diag_i[i] = G_diag_i[i-1] + counts[i-1];
      for (i = 0; i < ncols_G; i++)
      {
         for (j = T_diag_i[i]; j < T_diag_i[i+1]; j++)
         {
            k = T_diag_j[j];
            offset = G_diag_i[k]++;
            G_diag_j[offset] = i;
         }
      }
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++) G_diag_i[i] = G_diag_i[i-1] + counts[i-1];
      free(counts);
   }

   /* form G transpose in special form (2 nodes per edge max) */

   GT_diag_mat = (int *) malloc(2 * ncols_G * sizeof(int));
   for (i = 0; i < 2 * ncols_G; i++) GT_diag_mat[i] = -1;
   for (i = 0; i < nrows_G; i++)
   {
      for (j = G_diag_i[i]; j < G_diag_i[i+1]; j++)
      {
         edge = G_diag_j[j];
         if (GT_diag_mat[edge*2] == -1) GT_diag_mat[edge*2] = i;
         else                           GT_diag_mat[edge*2+1] = i;
      }
   }

   /* BFS on the local matrix graph to find tree */

   nodes_marked = (int *) malloc(nrows_G * sizeof(int));
   edges_marked = (int *) malloc(ncols_G * sizeof(int));
   for (i = 0; i < nrows_G; i++) nodes_marked[i] = 0; 
   for (i = 0; i < ncols_G; i++) edges_marked[i] = 0; 
   queue = (int *) malloc(nrows_G * sizeof(int));
   queue_head = 0;
   queue_tail = 1;
   queue[0] = 0;
   nodes_marked[0] = 1;
   while ((queue_tail-queue_head) > 0)
   {
      node = queue[queue_tail-1];
      queue_tail--;
      for (i = G_diag_i[node]; i < G_diag_i[node+1]; i++)
      {
         edge = G_diag_j[i]; 
         if (edges_marked[edge] == 0)
         {
            if (GT_diag_mat[2*edge+1] != -1)
            {
               node2 = GT_diag_mat[2*edge];
               if (node2 == node) node2 = GT_diag_mat[2*edge+1];
               if (nodes_marked[node2] == 0)
               {
                  nodes_marked[node2] = 1;
                  edges_marked[edge] = 1;
                  queue[queue_tail] = node2;
                  queue_tail++;
               }
            }
         }
      }
   }
   free(nodes_marked);
   free(queue);
   free(GT_diag_mat);

   /* fetch the communication information from */

   comm = hypre_ParCSRMatrixComm(G_csr);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   comm_pkg = hypre_ParCSRMatrixCommPkg(G_csr);
   if (nprocs == 1 && comm_pkg == NULL)
   {
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) G_csr);
      comm_pkg = hypre_ParCSRMatrixCommPkg(G_csr);
   }

   /* construct processor graph based on node-edge connection */
   /* (local edges connected to neighbor processor nodes)     */

   n_children = 0;
   nrecvs = nsends = 0;
   if (nprocs > 1)
   {
      nsends     = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
      nrecvs     = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      proc_array = NULL;
      if ((nsends+nrecvs) > 0)
      {
         n_proc_array = 0;
         proc_array = (int *) malloc((nsends+nrecvs) * sizeof(int));
         for (i = 0; i < nsends; i++) proc_array[i] = send_procs[i];
         for (i = 0; i < nrecvs; i++) proc_array[nsends+i] = recv_procs[i];
         qsort0(proc_array, 0, nsends+nrecvs-1); 
         n_proc_array = 1;
         for (i = 1; i < nrecvs+nsends; i++) 
            if (proc_array[i] != proc_array[n_proc_array])
               proc_array[n_proc_array++] = proc_array[i];
      }
      pgraph_i = (int *) malloc((nprocs+1) * sizeof(int));
      recv_cnts = (int *) malloc(nprocs * sizeof(int));
      MPI_Allgather(&n_proc_array, 1, MPI_INT, recv_cnts, 1, MPI_INT, comm);
      pgraph_i[0] = 0;
      for (i = 1; i <= nprocs; i++)
         pgraph_i[i] = pgraph_i[i-1] + recv_cnts[i-1];
      pgraph_j = (int *) malloc(pgraph_i[nprocs] * sizeof(int));
      MPI_Allgatherv(proc_array, n_proc_array, MPI_INT, pgraph_j, recv_cnts, 
                     pgraph_i, MPI_INT, comm);
      free(recv_cnts);

      /* BFS on the processor graph to determine parent and children */

      nodes_marked = (int *) malloc(nprocs * sizeof(int));
      for (i = 0; i < nprocs; i++) nodes_marked[i] = -1; 
      queue = (int *) malloc(nprocs * sizeof(int));
      queue_head = 0;
      queue_tail = 1;
      node = 0;
      queue[0] = node;
      while ((queue_tail-queue_head) > 0)
      {
         proc = queue[queue_tail-1];
         queue_tail--;
         for (i = pgraph_i[proc]; i < pgraph_i[proc+1]; i++)
         {
            proc2 = pgraph_j[i]; 
            if (nodes_marked[proc2] < 0)
            {
               nodes_marked[proc2] = proc;
               queue[queue_tail] = proc2;
               queue_tail++;
            }
         }
      }
      parent = nodes_marked[mypid];
      n_children = 0;
      for (i = 0; i < nprocs; i++) if (nodes_marked[i] == mypid) n_children++;
      if (n_children == 0) {n_children = 0; children = NULL;}
      else
      {
         children = (int *) malloc(n_children * sizeof(int));
         n_children = 0;
         for (i = 0; i < nprocs; i++) 
            if (nodes_marked[i] == mypid) children[n_children++] = i;
      } 
      free(nodes_marked);
      free(queue);
      free(pgraph_i);
      free(pgraph_j);
   }

   /* first, connection with my parent : if the edge in my parent *
    * is incident to one of my nodes, then my parent will mark it */

   found = 0;
   for (i = 0; i < nrecvs; i++)
   {
      proc = hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (proc == parent)
      {
         found = 1;
         break;
      }
   }

   /* but if all the edges connected to my parent are on my side, *
    * then I will just pick one of them as tree edge              */

   if (found == 0)
   {
      for (i = 0; i < nsends; i++)
      {
         proc = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == parent)
         {
            k = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
            edge = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }
   
   /* next, if my processor has an edge incident on one node in my *
    * child, put this edge on the tree. But if there is no such    *
    * edge, then I will assume my child will pick up an edge       */

   for (j = 0; j < n_children; i++)
   {
      proc = children[j];
      for (i = 0; i < nsends; i++)
      {
         proc2 = hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == proc2)
         {
            k = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
            edge = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }
   if (n_children > 0) free(children);

   /* count the size of the tree */

   tree_size = 0;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) tree_size++;
   t_indices = (int *) malloc((tree_size+1) * sizeof(int));
   t_indices[0] = tree_size;
   tree_size = 1;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) t_indices[tree_size++] = i;
   (*indices) = t_indices;
   free(edges_marked);
   if (G_type != 0)
   {
      free(G_diag_i);
      free(G_diag_j);
   }
}

/* -----------------------------------------------------------------------------
 * extract submatrices based on given indices
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixExtractSubmatrices(hypre_ParCSRMatrix *A_csr, int *indices2,
                                          hypre_ParCSRMatrix ***submatrices)
{
   int    nindices, *indices, nrows_A, *A_diag_i, *A_diag_j, mypid, nprocs;
   int    i, j, k, *proc_offsets1, *proc_offsets2, *itmp_array, *exp_indices;
   int    nnz11, nnz12, nnz21, nnz22, col, ncols_offd, nnz_offd, nnz_diag;
   int    global_nrows, global_ncols, *row_starts, *col_starts, nrows, nnz;
   int    *diag_i, *diag_j, row, *offd_i;
   double *A_diag_a, *diag_a;
   hypre_ParCSRMatrix *A11_csr, *A12_csr, *A21_csr, *A22_csr;
   hypre_CSRMatrix    *A_diag, *diag, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   qsort0(indices, 0, nindices-1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = hypre_CSRMatrixI(A_diag);
   A_diag_j = hypre_CSRMatrixJ(A_diag);
   A_diag_a = hypre_CSRMatrixData(A_diag);
   comm = hypre_ParCSRMatrixComm(A_csr);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   if (nprocs > 1)
   {
      printf("ExtractSubmatrices: cannot handle nprocs > 1 yet.\n");
      exit(1);
   }

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = (int *) malloc((nprocs+1) * sizeof(int));
   proc_offsets2 = (int *) malloc((nprocs+1) * sizeof(int));
   MPI_Allgather(&nindices, 1, MPI_INT, proc_offsets1, 1, MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++) 
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   } 
   proc_offsets1[nprocs] = k;
   itmp_array = hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++) 
      proc_offsets2[i] = itmp_array[i] - proc_offsets1[i];

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = (int *) malloc(nrows_A * sizeof(int));
   for (i = 0; i < nrows_A; i++) exp_indices[i] = -1;
   for (i = 0; i < nindices; i++) 
   {
      if (exp_indices[indices[i]] == -1) exp_indices[indices[i]] = i;
      else
      {
         printf("ExtractSubmatrices: wrong index %d %d\n", i, indices[i]);
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++) 
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz12 = nnz21 = nnz22 = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) nnz11++;
            else                       nnz12++;
         }
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) nnz21++;
            else                       nnz22++;
         }
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz11;
#ifdef HYPRE_NO_GLOBAL_PARTITION


#else
   global_nrows = proc_offsets1[nprocs];
   global_ncols = proc_offsets1[nprocs];
   row_starts = hypre_CTAlloc(int, nprocs+1);
   col_starts = hypre_CTAlloc(int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets1[i];
      col_starts[i] = proc_offsets1[i];
   }
#endif
   A11_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                    row_starts, col_starts, ncols_offd, nnz_diag, nnz_offd); 
   nrows = nindices;
   diag_i = hypre_CTAlloc(int, nrows+1);
   diag_j = hypre_CTAlloc(int, nnz_diag);
   diag_a = hypre_CTAlloc(double, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A11_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A11_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A12 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz12;
   global_nrows = proc_offsets1[nprocs];
   global_ncols = proc_offsets2[nprocs];
   row_starts = hypre_CTAlloc(int, nprocs+1);
   col_starts = hypre_CTAlloc(int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets1[i];
      col_starts[i] = proc_offsets2[i];
   }
   A12_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                    row_starts, col_starts, ncols_offd, nnz_diag, nnz_offd); 
   nrows = nindices;
   diag_i = hypre_CTAlloc(int, nrows+1);
   diag_j = hypre_CTAlloc(int, nnz_diag);
   diag_a = hypre_CTAlloc(double, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   if (nnz > nnz_diag) printf("WARNING WARNING WARNING\n");
   diag = hypre_ParCSRMatrixDiag(A12_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A12_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A21 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz21;
   global_nrows = proc_offsets2[nprocs];
   global_ncols = proc_offsets1[nprocs];
   row_starts = hypre_CTAlloc(int, nprocs+1);
   col_starts = hypre_CTAlloc(int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets2[i];
      col_starts[i] = proc_offsets1[i];
   }
   A21_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                    row_starts, col_starts, ncols_offd, nnz_diag, nnz_offd); 
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(int, nrows+1);
   diag_j = hypre_CTAlloc(int, nnz_diag);
   diag_a = hypre_CTAlloc(double, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A21_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A21_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A22 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz22;
   global_nrows = proc_offsets2[nprocs];
   global_ncols = proc_offsets2[nprocs];
   row_starts = hypre_CTAlloc(int, nprocs+1);
   col_starts = hypre_CTAlloc(int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets2[i];
      col_starts[i] = proc_offsets2[i];
   }
   A22_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                    row_starts, col_starts, ncols_offd, nnz_diag, nnz_offd); 
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(int, nrows+1);
   diag_j = hypre_CTAlloc(int, nnz_diag);
   diag_a = hypre_CTAlloc(double, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A22_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(int, nrows+1);
   for (i = 0; i <= nrows; i++) offd_i[i] = 0;
   offd = hypre_ParCSRMatrixOffd(A22_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = NULL;
   hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up 
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A12_csr;
   (*submatrices)[2] = A21_csr;
   (*submatrices)[3] = A22_csr;
   free(proc_offsets1);
   free(proc_offsets2);
   free(exp_indices);
}

/* -----------------------------------------------------------------------------
 * extract submatrices of a rectangular matrix
 * ----------------------------------------------------------------------------- */

void hypre_ParCSRMatrixExtractRowSubmatrices(hypre_ParCSRMatrix *A_csr, int *indices2,
                                             hypre_ParCSRMatrix ***submatrices)
{
   int    nindices, *indices, nrows_A, *A_diag_i, *A_diag_j, mypid, nprocs;
   int    i, j, k, *proc_offsets1, *proc_offsets2, *itmp_array, *exp_indices;
   int    nnz11, nnz21, col, ncols_offd, nnz_offd, nnz_diag, *A_offd_i, *A_offd_j;
   int    global_nrows, global_ncols, *row_starts, *col_starts, nrows, nnz;
   int    *diag_i, *diag_j, row, *offd_i, *offd_j, nnz11_offd, nnz21_offd;
   double *A_diag_a, *diag_a, *A_offd_a, *offd_a;
   hypre_ParCSRMatrix *A11_csr, *A21_csr;
   hypre_CSRMatrix    *A_diag, *diag, *A_offd, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   qsort0(indices, 0, nindices-1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = hypre_CSRMatrixI(A_diag);
   A_diag_j = hypre_CSRMatrixJ(A_diag);
   A_diag_a = hypre_CSRMatrixData(A_diag);
   A_offd = hypre_ParCSRMatrixOffd(A_csr);
   A_offd_i = hypre_CSRMatrixI(A_offd);
   A_offd_j = hypre_CSRMatrixJ(A_offd);
   A_offd_a = hypre_CSRMatrixData(A_offd);
   comm = hypre_ParCSRMatrixComm(A_csr);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = (int *) malloc((nprocs+1) * sizeof(int));
   proc_offsets2 = (int *) malloc((nprocs+1) * sizeof(int));
   MPI_Allgather(&nindices, 1, MPI_INT, proc_offsets1, 1, MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++) 
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   } 
   proc_offsets1[nprocs] = k;
   itmp_array = hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++) 
      proc_offsets2[i] = itmp_array[i] - proc_offsets1[i];

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = (int *) malloc(nrows_A * sizeof(int));
   for (i = 0; i < nrows_A; i++) exp_indices[i] = -1;
   for (i = 0; i < nindices; i++) 
   {
      if (exp_indices[indices[i]] == -1) exp_indices[indices[i]] = i;
      else
      {
         printf("ExtractRowSubmatrices: wrong index %d %d\n", i, indices[i]);
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++) 
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz21 = nnz11_offd = nnz21_offd = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) nnz11++;
         }
         nnz11_offd += A_offd_i[i+1] - A_offd_i[i];
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0) nnz21++;
         }
         nnz21_offd += A_offd_i[i+1] - A_offd_i[i];
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A_csr));
   nnz_diag   = nnz11;
   nnz_offd   = nnz11_offd; 

   global_nrows = proc_offsets1[nprocs];
   itmp_array   = hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = hypre_CTAlloc(int, nprocs+1);
   col_starts = hypre_CTAlloc(int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets1[i];
      col_starts[i] = itmp_array[i];
   }
   A11_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                    row_starts, col_starts, ncols_offd, nnz_diag, nnz_offd); 
   nrows = nindices;
   diag_i = hypre_CTAlloc(int, nrows+1);
   diag_j = hypre_CTAlloc(int, nnz_diag);
   diag_a = hypre_CTAlloc(double, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A11_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(int, nrows+1);
   offd_j = hypre_CTAlloc(int, nnz_offd);
   offd_a = hypre_CTAlloc(double, nnz_offd);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = hypre_ParCSRMatrixOffd(A11_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = offd_j;
   hypre_CSRMatrixData(offd) = offd_a;

   /* -----------------------------------------------------
    * create A21 matrix
    * ----------------------------------------------------- */

   ncols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A_csr));
   nnz_offd   = nnz21_offd;
   nnz_diag   = nnz21;
   global_nrows = proc_offsets2[nprocs];
   itmp_array   = hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = hypre_CTAlloc(int, nprocs+1);
   col_starts = hypre_CTAlloc(int, nprocs+1);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = proc_offsets2[i];
      col_starts[i] = itmp_array[i];
   }
   A21_csr = hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                    row_starts, col_starts, ncols_offd, nnz_diag, nnz_offd); 
   nrows = nrows_A - nindices;
   diag_i = hypre_CTAlloc(int, nrows+1);
   diag_j = hypre_CTAlloc(int, nnz_diag);
   diag_a = hypre_CTAlloc(double, nnz_diag);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            diag_j[nnz] = A_diag_j[j];
            diag_a[nnz++] = A_diag_a[j];
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = hypre_ParCSRMatrixDiag(A21_csr);
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixJ(diag) = diag_j;
   hypre_CSRMatrixData(diag) = diag_a;

   offd_i = hypre_CTAlloc(int, nrows+1);
   offd_j = hypre_CTAlloc(int, nnz_offd);
   offd_a = hypre_CTAlloc(double, nnz_offd);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i+1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = hypre_ParCSRMatrixOffd(A21_csr);
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixJ(offd) = offd_j;
   hypre_CSRMatrixData(offd) = offd_a;

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up 
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A21_csr;
   free(proc_offsets1);
   free(proc_offsets2);
   free(exp_indices);
}

