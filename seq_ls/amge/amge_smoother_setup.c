/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/






#include "headers.h"  

/*****************************************************************************
 *
 * builds smoother: LD and U arrays based on ILU(1) factorization of
 *                  Matrix in nested dissection ordering;
 *
 ****************************************************************************/


HYPRE_Int hypre_AMGeSmootherSetup(HYPRE_Int ***i_ILUdof_to_dof_pointer,

			    HYPRE_Int ***i_ILUdof_ILUdof_pointer,
			    HYPRE_Int ***j_ILUdof_ILUdof_pointer,
			    double ***LD_data_pointer,
			     
			    HYPRE_Int ***i_ILUdof_ILUdof_t_pointer,
			    HYPRE_Int ***j_ILUdof_ILUdof_t_pointer,
			    double ***U_data_pointer,


			    hypre_CSRMatrix **Matrix,

			    HYPRE_Int *level_pointer,

			    HYPRE_Int **i_block_node, HYPRE_Int **j_block_node,

			    HYPRE_Int **i_node_dof, HYPRE_Int **j_node_dof,

			    HYPRE_Int *Num_blocks, 
			    HYPRE_Int *Num_nodes,
			    HYPRE_Int *Num_dofs)

{
  HYPRE_Int ierr = 0;

  HYPRE_Int l;
  HYPRE_Int level = level_pointer[0];
  HYPRE_Int **i_ILUdof_to_dof;
  HYPRE_Int **i_ILUdof_ILUdof, **j_ILUdof_ILUdof;
  double **LD_data;
			     
  HYPRE_Int **i_ILUdof_ILUdof_t, **j_ILUdof_ILUdof_t;
  double **U_data;

  i_ILUdof_to_dof = hypre_CTAlloc(HYPRE_Int*, level);

  i_ILUdof_ILUdof = hypre_CTAlloc(HYPRE_Int*, level);
  j_ILUdof_ILUdof = hypre_CTAlloc(HYPRE_Int*, level);
  LD_data = hypre_CTAlloc(double*, level);

  i_ILUdof_ILUdof_t = hypre_CTAlloc(HYPRE_Int*, level);
  j_ILUdof_ILUdof_t = hypre_CTAlloc(HYPRE_Int*, level);
  U_data = hypre_CTAlloc(double*, level);


  l=0;
factorization_step:
  hypre_printf("\n\nB U I L D I N G  level[%d] ILU(1)  FACTORIZATION  M A T R I X\n",l);
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


  hypre_printf("LD_nnz: %d\n", i_ILUdof_ILUdof[l][Num_dofs[l]]);
  hypre_printf("U_nnz: %d\n", i_ILUdof_ILUdof_t[l][Num_dofs[l]]);
  hypre_printf("\n\n END building ILU(1) FACTORIZATION  MATRIX;------------------------\n");


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


 
