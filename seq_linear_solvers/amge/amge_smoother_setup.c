/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/


#include "headers.h"  

/*****************************************************************************
 *
 * builds smoother: LD and U arrays based on ILU(1) factorization of
 *                  Matrix in nested dissection ordering;
 *
 ****************************************************************************/


int hypre_AMGeSmootherSetup(int ***i_ILUdof_to_dof_pointer,

			    int ***i_ILUdof_ILUdof_pointer,
			    int ***j_ILUdof_ILUdof_pointer,
			    double ***LD_data_pointer,
			     
			    int ***i_ILUdof_ILUdof_t_pointer,
			    int ***j_ILUdof_ILUdof_t_pointer,
			    double ***U_data_pointer,


			    hypre_CSRMatrix **Matrix,

			    int *level_pointer,

			    int **i_block_node, int **j_block_node,

			    int **i_node_dof, int **j_node_dof,

			    int *Num_blocks, 
			    int *Num_nodes,
			    int *Num_dofs)

{
  int ierr = 0;

  int l;
  int level = level_pointer[0];
  int **i_ILUdof_to_dof;
  int **i_ILUdof_ILUdof, **j_ILUdof_ILUdof;
  double **LD_data;
			     
  int **i_ILUdof_ILUdof_t, **j_ILUdof_ILUdof_t;
  double **U_data;

  i_ILUdof_to_dof = hypre_CTAlloc(int*, level);

  i_ILUdof_ILUdof = hypre_CTAlloc(int*, level);
  j_ILUdof_ILUdof = hypre_CTAlloc(int*, level);
  LD_data = hypre_CTAlloc(double*, level);

  i_ILUdof_ILUdof_t = hypre_CTAlloc(int*, level);
  j_ILUdof_ILUdof_t = hypre_CTAlloc(int*, level);
  U_data = hypre_CTAlloc(double*, level);


  l=0;
factorization_step:
  printf("\n\nB U I L D I N G  level[%d] ILU(1)  FACTORIZATION  M A T R I X\n",l);
  ierr = hypre_ILUfactor(&i_ILUdof_to_dof[l],


			 &i_ILUdof_ILUdof[l],
			 &j_ILUdof_ILUdof[l],
			 &LD_data[l],
			     
			 &i_ILUdof_ILUdof_t[l],
			 &j_ILUdof_ILUdof_t[l],
			 &U_data[l],

			 Matrix[l],

			 i_node_dof[l], j_node_dof[l],

			 i_block_node[l], j_block_node[l],
			 Num_blocks[l], 
				    
			 Num_dofs[l],
			 Num_nodes[l]);


  printf("LD_nnz: %d\n", i_ILUdof_ILUdof[l][Num_dofs[l]]);
  printf("U_nnz: %d\n", i_ILUdof_ILUdof_t[l][Num_dofs[l]]);
  printf("\n\n END building ILU(1) FACTORIZATION  MATRIX;------------------------\n");


  /*
  hypre_TFree(i_block_node[l]);
  hypre_TFree(j_block_node[l]);


  if (l > 0)
    {
      hypre_TFree(i_node_dof[l]);
      hypre_TFree(j_node_dof[l]);
    }
  
  */

  l++;
  
  if (l < level && Num_nodes[l] > 0) goto factorization_step;
 
  level = l;


  *i_ILUdof_to_dof_pointer = i_ILUdof_to_dof;


  *i_ILUdof_ILUdof_pointer = i_ILUdof_ILUdof;
  *j_ILUdof_ILUdof_pointer = j_ILUdof_ILUdof;
  *LD_data_pointer = LD_data;
			     
  *i_ILUdof_ILUdof_t_pointer = i_ILUdof_ILUdof_t;
  *j_ILUdof_ILUdof_t_pointer = j_ILUdof_ILUdof_t;
  *U_data_pointer = U_data;

  return ierr;

}


 
