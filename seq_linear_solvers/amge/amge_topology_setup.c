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
 * builds AMGe matrix topology from element_node graph;
 *
 *
 ****************************************************************************/

int hypre_AMGeMatrixTopologySetup(hypre_AMGeMatrixTopology ***A_pointer,
				  int *level_pointer, 

				  int *i_element_node,
				  int *j_element_node,

				  int num_elements, int num_nodes, 

				  int Max_level)


{
  int ierr = 0;

  int num_faces;

  int level;

  int num_AEs;

  int *i_face_to_prefer_weight, *i_face_weight;
  hypre_AMGeMatrixTopology **A;

  int *i_boundarysurface_node=NULL, *j_boundarysurface_node=NULL;
  int num_boundarysurfaces = 0;


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

  i_face_to_prefer_weight = hypre_CTAlloc(int, num_faces);
  i_face_weight = hypre_CTAlloc(int, num_faces);

  level = 0;


agglomerate:
  A[level+1] = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1); 

  ierr = hypre_CoarsenAMGeMatrixTopology(A[level+1],

				    
					 A[level],
				    
					 i_face_to_prefer_weight,
					 i_face_weight);


  num_AEs = hypre_AMGeMatrixTopologyNumElements(A[level+1]);

  printf("level %d num_AEs: %d\n\n\n", level+1, num_AEs);

  level++;
   
  if (num_AEs > 1 && level+1 < Max_level) 
    goto agglomerate;

  printf("\n================================================================\n");
  printf("number of grids: from 0 to %d\n\n\n", level);
  printf("================================================================\n\n");


  hypre_TFree(i_face_to_prefer_weight);
  hypre_TFree(i_face_weight);

  *level_pointer = level;
  *A_pointer = A;



  return ierr;

}
