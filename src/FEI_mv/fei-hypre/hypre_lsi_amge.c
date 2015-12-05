/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_LSI_AMGE interface
 *
 *****************************************************************************/

#ifdef HAVE_AMGE

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "seq_ls/amge/AMGe_matrix_topology.h"
#include "seq_mv/csr_matrix.h"

extern int hypre_AMGeMatrixTopologySetup(hypre_AMGeMatrixTopology ***A,
                 int *level, int *i_element_node_0, int *j_element_node_0, 
                 int num_elements, int num_nodes, int Max_level);
extern int hypre_AMGeCoarsenodeSetup(hypre_AMGeMatrixTopology **A, int *level, 
                 int **i_node_neighbor_coarsenode, int **j_node_neighbor_coarsenode, 
                 int **i_node_coarsenode, int **j_node_coarsenode, 
                 int **i_block_node, int **j_block_node, int *Num_blocks, 
                 int *Num_elements, int *Num_nodes);

/* ********************************************************************* */
/* local variables to this module                                        */
/* ********************************************************************* */

int    rowLeng=0;
int    *i_element_node_0;
int    *j_element_node_0;
int    num_nodes, num_elements;
int    *i_dof_on_boundary;
int    system_size=1, num_dofs;
int    element_count=0;
int    temp_elemat_cnt;
int    **temp_elem_node, *temp_elem_node_cnt;
double **temp_elem_data;

/* ********************************************************************* */
/* constructor                                                           */
/* ********************************************************************* */

int HYPRE_LSI_AMGeCreate()
{
   printf("LSI_AMGe constructor\n");
   i_element_node_0   = NULL;
   j_element_node_0   = NULL;
   num_nodes          = 0;
   num_elements       = 0;
   system_size        = 1;
   num_dofs           = 0;
   element_count      = 0;
   temp_elemat_cnt    = 0;
   temp_elem_node     = NULL;
   temp_elem_node_cnt = NULL;
   temp_elem_data     = NULL;
   i_dof_on_boundary  = NULL;
   return 0;
}

/* ********************************************************************* */
/* destructor                                                            */
/* ********************************************************************* */

int HYPRE_LSI_AMGeDestroy()
{
   int i;

   printf("LSI_AMGe destructor\n");
   if ( i_element_node_0   != NULL ) free( i_element_node_0 );
   if ( j_element_node_0   != NULL ) free( j_element_node_0 );
   if ( i_dof_on_boundary  != NULL ) free( i_dof_on_boundary );
   if ( temp_elem_node_cnt != NULL ) free( temp_elem_node_cnt );
   for ( i = 0; i < num_elements; i++ )
   {
      if ( temp_elem_node[i] != NULL ) free( temp_elem_node[i] );
      if ( temp_elem_data[i] != NULL ) free( temp_elem_data[i] );
   }
   temp_elem_node     = NULL;
   temp_elem_node_cnt = NULL;
   temp_elem_data     = NULL;
   return 0;
}

/* ********************************************************************* */
/* set the number of nodes in the finest grid                            */
/* ********************************************************************* */

int HYPRE_LSI_AMGeSetNNodes(int nNodes)
{
   int i;

   printf("LSI_AMGe NNodes = %d\n", nNodes);
   num_nodes = nNodes;
   return 0;
}

/* ********************************************************************* */
/* set the number of elements in the finest grid                         */
/* ********************************************************************* */

int HYPRE_LSI_AMGeSetNElements(int nElems)
{
   int i, nbytes;

   printf("LSI_AMGe NElements = %d\n", nElems);
   num_elements = nElems;
   nbytes = num_elements * sizeof(double*);
   temp_elem_data = (double **) malloc( nbytes );
   for ( i = 0; i < num_elements; i++ ) temp_elem_data[i] = NULL;
   nbytes = num_elements * sizeof(int*);
   temp_elem_node = (int **) malloc( nbytes );
   for ( i = 0; i < num_elements; i++ ) temp_elem_node[i] = NULL;
   nbytes = num_elements * sizeof(int);
   temp_elem_node_cnt = (int *) malloc( nbytes );
   return 0;
}

/* ********************************************************************* */
/* set system size                                                       */
/* ********************************************************************* */

int HYPRE_LSI_AMGeSetSystemSize(int size)
{
   printf("LSI_AMGe SystemSize = %d\n", size);
   system_size = size;
   return 0;
}

/* ********************************************************************* */
/* set boundary condition                                                */
/* ********************************************************************* */

int HYPRE_LSI_AMGeSetBoundary(int size, int *list)
{
   int i;

   printf("LSI_AMGe SetBoundary = %d\n", size);

   if ( i_dof_on_boundary == NULL )
      i_dof_on_boundary = (int *) malloc(num_nodes * system_size * sizeof(int));
   for ( i = 0; i < num_nodes*system_size; i++ ) i_dof_on_boundary[i] = -1;

   for ( i = 0; i < size; i++ ) 
   {
      if (list[i] >= 0 && list[i] < num_nodes*system_size) 
         i_dof_on_boundary[list[i]] = 0;
      else printf("AMGeSetBoundary ERROR : %d(%d)\n", list[i],num_nodes*system_size);
   }
   return 0;
}

/* ********************************************************************* */
/* load a row into this module                                           */
/* ********************************************************************* */

int HYPRE_LSI_AMGePutRow(int row, int length, const double *colVal,
                          const int *colInd)
{
   int i, nbytes;

   if ( rowLeng == 0 )
   {
      if ( element_count % 100 == 0 )
         printf("LSI_AMGe PutRow %d\n", element_count);
      if ( element_count < 0 || element_count >= num_elements )
         printf("ERROR : element count too large %d\n",element_count);

      temp_elem_node_cnt[element_count] = length / system_size;
      nbytes = length / system_size * sizeof(int);
      temp_elem_node[element_count] = (int *) malloc( nbytes );
      for ( i = 0; i < length; i+=system_size ) 
         temp_elem_node[element_count][i/system_size] = (colInd[i]-1)/system_size;
      nbytes = length * length * sizeof(double);
      temp_elem_data[element_count] = (double *) malloc(nbytes);
      temp_elemat_cnt = 0;
      rowLeng = length;
   }
   for ( i = 0; i < length; i++ ) 
      temp_elem_data[element_count][temp_elemat_cnt++] = colVal[i];
   if ( temp_elemat_cnt == rowLeng * rowLeng )
   {
      element_count++;
      rowLeng = 0;
   }
   return 0;
}

/* ********************************************************************* */
/* Solve                                                                 */
/* ********************************************************************* */

int HYPRE_LSI_AMGeSolve(double *rhs, double *x)
{
   int    i, j, l, counter, ierr, total_length;
   int    *Num_nodes, *Num_elements, *Num_dofs, level;
   int    max_level, Max_level;
   int    multiplier;

   /* coarsenode information and coarsenode neighborhood information */

   int **i_node_coarsenode, **j_node_coarsenode;
   int **i_node_neighbor_coarsenode, **j_node_neighbor_coarsenode;

   /* PDEsystem information: --------------------------------------- */

   int *i_dof_node_0, *j_dof_node_0;
   int *i_node_dof_0, *j_node_dof_0;

   int *i_element_dof_0, *j_element_dof_0;
   double *element_data;

   int **i_node_dof, **j_node_dof;

   /* Dirichlet boundary conditions information: ------------------- */

   /* int *i_dof_on_boundary; */

   /* nested dissection blocks: ------------------------------------ */

   int **i_block_node, **j_block_node;
   int *Num_blocks;

   /* nested dissection ILU(1) smoother: --------------------------- */
   /* internal format: --------------------------------------------- */

   int **i_ILUdof_to_dof;
   int **i_ILUdof_ILUdof_t, **j_ILUdof_ILUdof_t,
       **i_ILUdof_ILUdof, **j_ILUdof_ILUdof;
   double **LD_data, **U_data;

   /* -------------------------------------------------------------- */
   /*  PCG & V_cycle arrays:                                         */
   /* -------------------------------------------------------------- */

   double *r, *v, **w, **d, *aux, *v_coarse, *w_coarse;
   double *d_coarse, *v_fine, *w_fine, *d_fine;
   int max_iter = 1000;
   int coarse_level;
   int nu = 1;  /* not used ---------------------------------------- */

   double reduction_factor;

   /* Interpolation P and stiffness matrices Matrix; --------------- */

   hypre_CSRMatrix     **P;
   hypre_CSRMatrix     **Matrix;
   hypre_AMGeMatrixTopology **A;

   /* element matrices information: -------------------------------- */

   int *i_element_chord_0, *j_element_chord_0;
   double *a_element_chord_0;
   int *i_chord_dof_0, *j_chord_dof_0;
   int *Num_chords;

   /* auxiliary arrays for enforcing Dirichlet boundary conditions:  */

   int *i_dof_dof_a, *j_dof_dof_a;
   double *a_dof_dof;

   /* ===============================================================*/
   /* set num_nodes, num_elements                                    */
   /* fill up element_data                                           */
   /* fill up i_element_node_0 and j_element_node_0                  */
   /* fill up i_dof_on_boundary (0 - boundary, 1 - otherwise)        */
   /* ===============================================================*/

   num_elements = element_count;
   if ( num_nodes == 0 || num_elements == 0 )
   {
      printf("HYPRE_LSI_AMGe ERROR : num_nodes or num_elements not set.\n");
      exit(1);
   }
   total_length = 0;
   for ( i = 0; i < num_elements; i++ )
   {
      multiplier = temp_elem_node_cnt[i] * system_size;
      total_length += (multiplier * multiplier);
   }
   element_data = (double *) malloc(total_length * sizeof(double));
   counter = 0;
   for ( i = 0; i < num_elements; i++ )
   {
      multiplier = temp_elem_node_cnt[i] * system_size;
      multiplier *= multiplier;
      for ( j = 0; j < multiplier; j++ )
         element_data[counter++] = temp_elem_data[i][j];
      free(temp_elem_data[i]);
   }  
   free(temp_elem_data);
   temp_elem_data = NULL;

   total_length = 0;
   for (i = 0; i < num_elements; i++) total_length += temp_elem_node_cnt[i];
   i_element_node_0 = (int *) malloc((num_elements + 1) * sizeof(int));
   j_element_node_0 = (int *) malloc(total_length * sizeof(int));
   counter = 0;
   for (i = 0; i < num_elements; i++) 
   {
      i_element_node_0[i] = counter;
      for (j = 0; j < temp_elem_node_cnt[i]; j++) 
         j_element_node_0[counter++] = temp_elem_node[i][j];
      free(temp_elem_node[i]);
   } 
   i_element_node_0[num_elements] = counter;
   free(temp_elem_node);
   temp_elem_node = NULL;

   /* -------------------------------------------------------------- */
   /* initialization                                                 */
   /* -------------------------------------------------------------- */

   Max_level    = 25;
   Num_chords   = hypre_CTAlloc(int, Max_level);
   Num_elements = hypre_CTAlloc(int, Max_level);
   Num_nodes    = hypre_CTAlloc(int, Max_level);
   Num_dofs     = hypre_CTAlloc(int, Max_level);
   Num_blocks   = hypre_CTAlloc(int, Max_level);

   for (i = 0; i < Max_level; i++)
   {
      Num_dofs[i] = 0;
      Num_elements[i] = 0;
   }

   Num_nodes[0] = num_nodes;
   Num_elements[0] = num_elements;

   /* -------------------------------------------------------------- */
   /* set up matrix topology for the fine matrix                     */
   /* input : i_element_node_0, j_element_node_0, num_elements,      */
   /*         num_nodes, Max_level                                   */
   /* -------------------------------------------------------------- */

   printf("LSI_AMGe Solve : Setting up topology \n");
   ierr = hypre_AMGeMatrixTopologySetup(&A, &level, i_element_node_0,
                j_element_node_0, num_elements, num_nodes, Max_level);

   max_level = level;

   /* -------------------------------------------------------------- */
   /* set up matrix topology for the coarse grids                    */
   /* input : A, Num_elements[0], Num_nodes[0]                       */
   /* -------------------------------------------------------------- */

   printf("LSI_AMGe Solve : Setting up coarse grids \n");
   ierr = hypre_AMGeCoarsenodeSetup(A, &level, &i_node_neighbor_coarsenode,
                &j_node_neighbor_coarsenode, &i_node_coarsenode,
                &j_node_coarsenode, &i_block_node, &j_block_node,
                Num_blocks, Num_elements, Num_nodes);

   /* -------------------------------------------------------------- */
   /* set up dof arrays based on system size                         */
   /* output : i_dof_node_0, j_dof_node_0, num_dofs                  */
   /* -------------------------------------------------------------- */

   ierr = compute_dof_node(&i_dof_node_0, &j_dof_node_0,
                           Num_nodes[0], system_size, &num_dofs);

   Num_dofs[0] = num_dofs;

   /*
   if (system_size == 1) i_dof_on_boundary = i_node_on_boundary;
   else
   {
      ierr = compute_dof_on_boundary(&i_dof_on_boundary, i_node_on_boundary,
                                     Num_nodes[0], system_size);
      free(i_node_on_boundary);
      i_node_on_boundary = NULL;
   }
   */

   /* -------------------------------------------------------------- */
   /* get element_dof information                                    */
   /* -------------------------------------------------------------- */

   ierr = transpose_matrix_create(&i_node_dof_0, &j_node_dof_0,
                   i_dof_node_0, j_dof_node_0, Num_dofs[0], Num_nodes[0]);

   if (system_size == 1)
   {
      i_element_dof_0 = i_element_node_0;
      j_element_dof_0 = j_element_node_0;
   }
   else
      ierr = matrix_matrix_product(&i_element_dof_0, &j_element_dof_0,
                i_element_node_0,j_element_node_0,i_node_dof_0,j_node_dof_0,
                Num_elements[0], Num_nodes[0], Num_dofs[0]);

   /* -------------------------------------------------------------- */
   /* store element matrices in element_chord format                 */
   /* -------------------------------------------------------------- */

   printf("LSI_AMGe Solve : Setting up element dof relations \n");
   ierr = hypre_AMGeElementMatrixDof(i_element_dof_0, j_element_dof_0,
                element_data, &i_element_chord_0, &j_element_chord_0,
                &a_element_chord_0, &i_chord_dof_0, &j_chord_dof_0,
                &Num_chords[0], Num_elements[0], Num_dofs[0]);

   printf("LSI_AMGe Solve : Setting up interpolation \n");
   ierr = hypre_AMGeInterpolationSetup(&P, &Matrix, A, &level,
                /* ------ fine-grid element matrices ----- */
                i_element_chord_0, j_element_chord_0, a_element_chord_0,
                i_chord_dof_0, j_chord_dof_0,

                /* nnz: of the assembled matrices -------*/
                Num_chords,

                /* ----- coarse node information  ------ */
                i_node_neighbor_coarsenode, j_node_neighbor_coarsenode,
                i_node_coarsenode, j_node_coarsenode,

                /* --------- Dirichlet b.c. ----------- */
                i_dof_on_boundary,

                /* -------- PDEsystem information -------- */
                system_size, i_dof_node_0, j_dof_node_0,
                i_node_dof_0, j_node_dof_0, &i_node_dof, &j_node_dof,

                Num_elements, Num_nodes, Num_dofs);

   hypre_TFree(i_dof_on_boundary);
   i_dof_on_boundary = NULL;
   hypre_TFree(i_dof_node_0);
   hypre_TFree(j_dof_node_0);

   printf("LSI_AMGe Solve : Setting up smoother \n");
   ierr = hypre_AMGeSmootherSetup(&i_ILUdof_to_dof, &i_ILUdof_ILUdof,
                &j_ILUdof_ILUdof, &LD_data, &i_ILUdof_ILUdof_t,
                &j_ILUdof_ILUdof_t, &U_data, Matrix, &level,
                i_block_node, j_block_node, i_node_dof, j_node_dof,
                Num_blocks, Num_nodes, Num_dofs);

   hypre_TFree(i_node_dof_0);
   hypre_TFree(j_node_dof_0);

   for (l=0; l < level+1; l++)
   {
      hypre_TFree(i_block_node[l]);
      hypre_TFree(j_block_node[l]);
   }

   for (l=1; l < level+1; l++)
   {
      hypre_TFree(i_node_dof[l]);
      hypre_TFree(j_node_dof[l]);
   }

   hypre_TFree(i_node_dof);
   hypre_TFree(j_node_dof);
   hypre_TFree(i_block_node);
   hypre_TFree(j_block_node);

   /* ===================================================================== */
   /* =================== S O L U T I O N   P A R T: ====================== */
   /* ===================================================================== */

   /* one V(1,1) --cycle as preconditioner in PCG: ======================== */
   /* ILU solve pre--smoothing, ILU solve post--smoothing; ================ */

   w = hypre_CTAlloc(double*, level+1); 
   d = hypre_CTAlloc(double*, level+1);

   for (l=0; l < level+1; l++)
   {
      Num_dofs[l] = Num_nodes[l] * system_size;
      if (Num_dofs[l] > 0)
      {
	  w[l] = hypre_CTAlloc(double, Num_dofs[l]);
	  d[l] = hypre_CTAlloc(double, Num_dofs[l]);
      }
      else
      {
	  level = l-1;
	  break;
      }
   }

   num_dofs = Num_dofs[0];

   /*x = hypre_CTAlloc(double, num_dofs);  */
   /*rhs = hypre_CTAlloc(double, num_dofs);*/

   r = hypre_CTAlloc(double, num_dofs); 
   aux = hypre_CTAlloc(double, num_dofs);
   v_fine = hypre_CTAlloc(double, num_dofs);
   w_fine = hypre_CTAlloc(double, num_dofs);
   d_fine = hypre_CTAlloc(double, num_dofs);

   coarse_level = level;
   v_coarse = hypre_CTAlloc(double, Num_dofs[coarse_level]);
   w_coarse = hypre_CTAlloc(double, Num_dofs[coarse_level]);
   d_coarse = hypre_CTAlloc(double, Num_dofs[coarse_level]);

   for (l=0; l < level; l++)
   {
      printf("\n\n=======================================================\n");
      printf("             Testing level[%d] PCG solve:                  \n",l);
      printf("===========================================================\n");
 
      for (i=0; i < Num_dofs[l]; i++) x[i] = 0.e0;

      /* for (i=0; i < Num_dofs[l]; i++) rhs[i] = rand(); */

      i_dof_dof_a = hypre_CSRMatrixI(Matrix[l]);
      j_dof_dof_a = hypre_CSRMatrixJ(Matrix[l]);
      a_dof_dof   = hypre_CSRMatrixData(Matrix[l]);

      ierr = hypre_ILUsolve(x, i_ILUdof_to_dof[l], i_ILUdof_ILUdof[l],
	           j_ILUdof_ILUdof[l], LD_data[l], i_ILUdof_ILUdof_t[l],
                   j_ILUdof_ILUdof_t[l], U_data[l], rhs, Num_dofs[l]);

      ierr = hypre_ILUpcg(x, rhs, a_dof_dof, i_dof_dof_a, j_dof_dof_a,
                   i_ILUdof_to_dof[l], i_ILUdof_ILUdof[l], j_ILUdof_ILUdof[l],
                   LD_data[l], i_ILUdof_ILUdof_t[l], j_ILUdof_ILUdof_t[l],
                   U_data[l], v_fine, w_fine, d_fine, max_iter, Num_dofs[l]);

      printf("\n\n=======================================================\n");
      printf("             END test PCG solve:                           \n");
      printf("===========================================================\n");
 
   }

   printf("\n\n===============================================================\n");
   printf(" ------- V_cycle & nested dissection ILU(1) smoothing: --------\n");
   printf("================================================================\n");

   num_dofs = Num_dofs[0];

   /* for (i=0; i < num_dofs; i++) rhs[i] = rand(); */
  
   ierr = hypre_VcycleILUpcg(x, rhs, w, d, &reduction_factor, Matrix,
                i_ILUdof_to_dof, i_ILUdof_ILUdof, j_ILUdof_ILUdof, LD_data,
                i_ILUdof_ILUdof_t, j_ILUdof_ILUdof_t, U_data, P, aux, r, 
                v_fine, w_fine, d_fine, max_iter, v_coarse, w_coarse, d_coarse, 
                nu, level, coarse_level, Num_dofs);

   /* hypre_TFree(x);   */
   /* hypre_TFree(rhs); */

   hypre_TFree(r);
   hypre_TFree(aux);

   for (l=0; l < level+1; l++)
      if (Num_dofs[l] > 0)
      {
	hypre_TFree(w[l]);
	hypre_TFree(d[l]);
	hypre_CSRMatrixDestroy(Matrix[l]);
      }

   for (l=0; l < max_level; l++)
   {
      hypre_TFree(i_node_coarsenode[l]);
      hypre_TFree(j_node_coarsenode[l]);

      hypre_TFree(i_node_neighbor_coarsenode[l]);
      hypre_TFree(j_node_neighbor_coarsenode[l]);

      if (system_size == 1 &&Num_dofs[l+1] > 0)
      {
	  hypre_CSRMatrixI(P[l]) = NULL;
	  hypre_CSRMatrixJ(P[l]) = NULL;
      }
  
   }
   for (l=0; l < level; l++)
   {
      hypre_TFree(i_ILUdof_to_dof[l]);
      hypre_TFree(i_ILUdof_ILUdof[l]);
      hypre_TFree(j_ILUdof_ILUdof[l]);
      hypre_TFree(LD_data[l]);
      hypre_TFree(i_ILUdof_ILUdof_t[l]);
      hypre_TFree(j_ILUdof_ILUdof_t[l]);
      hypre_TFree(U_data[l]);
      hypre_CSRMatrixDestroy(P[l]);

   }

   hypre_TFree(v_fine);
   hypre_TFree(w_fine);
   hypre_TFree(d_fine);
   hypre_TFree(w);
   hypre_TFree(d);

   hypre_TFree(v_coarse);
   hypre_TFree(w_coarse);
   hypre_TFree(d_coarse);

   for (l=0; l < max_level+1; l++)
      hypre_DestroyAMGeMatrixTopology(A[l]);

   hypre_TFree(Num_nodes);
   hypre_TFree(Num_elements);
   hypre_TFree(Num_dofs);
   hypre_TFree(Num_blocks);
   hypre_TFree(Num_chords);

   hypre_TFree(i_chord_dof_0);
   hypre_TFree(j_chord_dof_0);

   hypre_TFree(i_element_chord_0);
   hypre_TFree(j_element_chord_0);
   hypre_TFree(a_element_chord_0);

   hypre_TFree(P);
   hypre_TFree(Matrix);
   hypre_TFree(A);

   hypre_TFree(i_ILUdof_to_dof);
   hypre_TFree(i_ILUdof_ILUdof);
   hypre_TFree(j_ILUdof_ILUdof);
   hypre_TFree(LD_data);

   hypre_TFree(i_ILUdof_ILUdof_t);
   hypre_TFree(j_ILUdof_ILUdof_t);
   hypre_TFree(U_data);

   hypre_TFree(i_node_coarsenode);
   hypre_TFree(j_node_coarsenode);

   hypre_TFree(i_node_neighbor_coarsenode);
   hypre_TFree(j_node_neighbor_coarsenode);
   free(element_data);

   return 0;
}

/* ********************************************************************* */
/* local variables to this module                                        */
/* ********************************************************************* */

int HYPRE_LSI_AMGeWriteToFile()
{
   int  i, j, k, length;
   FILE *fp;

   fp = fopen("elem_mat", "w");

   for ( i = 0; i < element_count; i++ )
   {
      length = temp_elem_node_cnt[i] * system_size;
      for ( j = 0; j < length; j++ )
      {
         for ( k = 0; k < length; k++ )
            fprintf(fp, "%13.6e ", temp_elem_data[i][j*length+k]);
         fprintf(fp, "\n");
      }
      fprintf(fp, "\n");
   }  
   fclose(fp);

   fp = fopen("elem_node", "w");
   
   fprintf(fp, "%d %d\n", element_count, num_nodes);
   for (i = 0; i < element_count; i++) 
   {
      for (j = 0; j < temp_elem_node_cnt[i]; j++) 
         fprintf(fp, "%d ", temp_elem_node[i][j]+1);
      fprintf(fp,"\n");
   } 

   fclose(fp);

   fp = fopen("node_bc", "w");

   for (i = 0; i < num_nodes*system_size; i++) 
   {
      fprintf(fp, "%d\n", i_dof_on_boundary[i]);
   }
   fclose(fp);

   return 0;
}

#else

/* this is used only to eliminate compiler warnings */
int hypre_empty4;

#endif

