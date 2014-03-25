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
 * builds AMGe matrix topology from element_node graph;
 *
 *
 ****************************************************************************/

HYPRE_Int hypre_AMGeMatrixTopologySetup(hypre_AMGeMatrixTopology ***A_pointer,
				  HYPRE_Int *level_pointer, 

				  HYPRE_Int *i_element_node,
				  HYPRE_Int *j_element_node,

				  HYPRE_Int num_elements, HYPRE_Int num_nodes, 

				  HYPRE_Int Max_level)


{
  HYPRE_Int ierr = 0;

  HYPRE_Int num_faces;

  HYPRE_Int level;

  HYPRE_Int num_AEs;

  HYPRE_Int *i_face_to_prefer_weight, *i_face_weight;
  hypre_AMGeMatrixTopology **A;

  HYPRE_Int *i_boundarysurface_node=NULL, *j_boundarysurface_node=NULL;
  HYPRE_Int num_boundarysurfaces = 0;


  A = hypre_CTAlloc(hypre_AMGeMatrixTopology*, Max_level);

  ierr = hypre_BuildAMGeMatrixTopology(&A[0],
				       i_element_node,
				       j_element_node,

				       i_boundarysurface_node,
				       j_boundarysurface_node,

				       num_elements,
				       num_nodes,
				       num_boundarysurfaces);



  num_faces = hypre_AMGeMatrixTopologyNumFaces(A[0]);

  i_face_to_prefer_weight = hypre_CTAlloc(HYPRE_Int, num_faces);
  i_face_weight = hypre_CTAlloc(HYPRE_Int, num_faces);

  level = 0;


agglomerate:
  A[level+1] = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1); 

  ierr = hypre_CoarsenAMGeMatrixTopology(A[level+1],

				    
					 A[level],
				    
					 i_face_to_prefer_weight,
					 i_face_weight);


  num_AEs = hypre_AMGeMatrixTopologyNumElements(A[level+1]);

  hypre_printf("level %d num_AEs: %d\n\n\n", level+1, num_AEs);

  level++;
   
  if (num_AEs > 1 && level+1 < Max_level) 
    goto agglomerate;

  hypre_printf("\n================================================================\n");
  hypre_printf("number of grids: from 0 to %d\n\n\n", level);
  hypre_printf("================================================================\n\n");


  hypre_TFree(i_face_to_prefer_weight);
  hypre_TFree(i_face_weight);

  *level_pointer = level;
  *A_pointer = A;



  return ierr;

}
