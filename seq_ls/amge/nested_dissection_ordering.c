/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/






#include "headers.h"  

/*****************************************************************************
 *
 * builds nested dissection ordering based on "node_level"
 *
 * i_node_level[i] = # faces
 * the node "i" belongs to;
 *
 * at all levels created throughout the AMGeMatrixTopology process; 
 *
 * if i_node_level = i_node_HB_level as input, the output
 *
 *           "level_node" 
 *
 *    is the HB node ordering;
 *
 ****************************************************************************/

int AMGeNestedDissectionOrdering(int *i_node_level,
				 int num_nodes, 
				 int level, 

				 int **j_node_level_pointer,

				 int **i_level_node_pointer, 
				 int **j_level_node_pointer, 

				 int *num_levels_pointer)
{
  int ierr = 0;
  int i,j,k,l;  

  int *j_node_level, *i_level_node, *j_level_node;

  int min_level, max_level;

  j_node_level = hypre_CTAlloc(int, num_nodes);

  max_level = 0;
  min_level = level;
  for (i=0; i < num_nodes; i++)
    {
      if (max_level < i_node_level[i]) 
	max_level = i_node_level[i];

      if (min_level > i_node_level[i]) 
	min_level = i_node_level[i];

    }

  printf("level: %d, max_level: %d, min_level: %d\n", level, max_level,
	 min_level);

  for (i=0; i < num_nodes; i++)
    {
      j_node_level[i] = i_node_level[i]-min_level;
      i_node_level[i] = i;
    }

  i_node_level[num_nodes] = num_nodes;


  ierr = transpose_matrix_create(&i_level_node, &j_level_node,

				 i_node_level, j_node_level,
				 num_nodes, max_level-min_level+1);


  *num_levels_pointer = max_level-min_level+1;
  *i_level_node_pointer = i_level_node;
  *j_level_node_pointer = j_level_node;

  *j_node_level_pointer = j_node_level; 


  printf("\n==============================================================\n");
  printf("\n     n e s t e d   d i s s e c t i o n   o r d e r i n g:     \n");
  printf("\n==============================================================\n");


  for (l=0; l < max_level-min_level+1; l++)
    {
      printf("level: %d contains %d nodes: \n", l, 
	     i_level_node[l+1]-i_level_node[l]);
      /*
      for (k=i_level_node[l]; k < i_level_node[l+1]; k++)
	printf(" %d, ", j_level_node[k]);
	*/
      printf("\n\n");
    }
  printf("\n==============================================================\n");
  printf("num_nodes %d and num_nodes counted: %d\n\n\n",
	 num_nodes, i_level_node[max_level-min_level+1]);


  return ierr;

}
