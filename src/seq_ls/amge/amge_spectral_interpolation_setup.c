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

#include "hypre_amge.h"

/*****************************************************************************
 *
 * builds interpolation matrices P, matrix topology A,
 *        and stiffness matrices Matrix;
 * input for initial matrix topology; element matrices, chords etc;
 *
 ****************************************************************************/


HYPRE_Int 
hypre_AMGeSpectralInterpolationSetup(hypre_CSRMatrix ***P_pointer,

			    hypre_CSRMatrix ***Matrix_pointer,

			    hypre_AMGeMatrixTopology **A, 


			    HYPRE_Int *level_pointer,

			    /* ------ fine-grid element matrices ----- */
			    HYPRE_Int *i_element_chord_0,
			    HYPRE_Int *j_element_chord_0,
			    HYPRE_Real *a_element_chord_0,

			    HYPRE_Int *i_chord_dof_0,
			    HYPRE_Int *j_chord_dof_0,

			    HYPRE_Int *i_element_dof_0,
			    HYPRE_Int *j_element_dof_0,

			    /* nnz: of the assembled matrices -------*/
			    HYPRE_Int *Num_chords,


			    /* --------- Dirichlet b.c. ----------- */
			    HYPRE_Int *i_dof_on_boundary, 

			    /* --------------------------------------- */

			    HYPRE_Real tolerance, 


			    HYPRE_Int *Num_elements,
			    HYPRE_Int *Num_dofs,

			    HYPRE_Int Max_level)

{
  HYPRE_Int ierr = 0;

  HYPRE_Int i,j, l;


  HYPRE_Int *i_dof_dof_a, *j_dof_dof_a;
  HYPRE_Real *a_dof_dof;

  HYPRE_Int *i_aggregate_dof, *j_aggregate_dof;
  HYPRE_Int *i_dof_aggregate, *j_dof_aggregate;

  HYPRE_Int *i_coarseaggregate_coarsedof, *j_coarseaggregate_coarsedof;


  hypre_CSRMatrix **Matrix, **P;

  HYPRE_Int level;

  HYPRE_Int num_faces;
  HYPRE_Int *i_AE_element, *j_AE_element;
  HYPRE_Int num_AEs;


  HYPRE_Int num_AEfaces;

  HYPRE_Int *i_face_to_prefer_weight, *i_face_weight;
  /* hypre_AMGeMatrixTopology **A; */

  HYPRE_Int *i_boundarysurface_dof=NULL, *j_boundarysurface_dof=NULL;
  HYPRE_Int num_boundarysurfaces = 0;


  HYPRE_Int **i_chord_dof, **j_chord_dof;
  HYPRE_Int **i_element_dof, **j_element_dof;
  HYPRE_Int **i_element_chord, **j_element_chord;
  HYPRE_Real **a_element_chord;

  HYPRE_Int **i_dof_coarsedof, **j_dof_coarsedof;

  HYPRE_Int *i_dof_element, *j_dof_element;



  i_chord_dof = hypre_CTAlloc(HYPRE_Int*, Max_level+1);
  j_chord_dof = hypre_CTAlloc(HYPRE_Int*, Max_level+1);

  i_chord_dof[0] = i_chord_dof_0;
  j_chord_dof[0] = j_chord_dof_0;

  i_element_dof = hypre_CTAlloc(HYPRE_Int*, Max_level+1);
  j_element_dof = hypre_CTAlloc(HYPRE_Int*, Max_level+1);

  i_element_dof[0] = i_element_dof_0;
  j_element_dof[0] = j_element_dof_0;

  i_element_chord = hypre_CTAlloc(HYPRE_Int*, Max_level+1);
  j_element_chord = hypre_CTAlloc(HYPRE_Int*, Max_level+1);
  a_element_chord = hypre_CTAlloc(HYPRE_Real*, Max_level+1);

  i_element_chord[0] = i_element_chord_0;
  j_element_chord[0] = j_element_chord_0;
  a_element_chord[0] = a_element_chord_0;

  i_dof_coarsedof = hypre_CTAlloc(HYPRE_Int*, Max_level);
  j_dof_coarsedof = hypre_CTAlloc(HYPRE_Int*, Max_level);



  Matrix = hypre_CTAlloc(hypre_CSRMatrix*, Max_level+1);
  P      = hypre_CTAlloc(hypre_CSRMatrix*, Max_level);

  /*
  A      = hypre_CTAlloc(hypre_AMGeMatrixTopology*, Max_level);


  ierr = hypre_BuildAMGeMatrixTopology(&A[0],
				       i_element_dof[0],
				       j_element_dof[0],

				       i_boundarysurface_dof,
				       j_boundarysurface_dof,

				       Num_elements[0],
				       Num_dofs[0],
				       num_boundarysurfaces);




  if (!hypre_AMGeMatrixTopologyIElementNode(A[0]))
    hypre_AMGeMatrixTopologyIElementNode(A[0]) = 
      i_element_dof[0];

  if (!hypre_AMGeMatrixTopologyJElementNode(A[0]))
    hypre_AMGeMatrixTopologyJElementNode(A[0]) = 
      j_element_dof[0];

  hypre_AMGeMatrixTopologyNumNodes(A[0]) = Num_dofs[0];
  */

  /* assemble initial fine matrix: ------------------------------------- */
  hypre_printf("Num_elements[0]: %d, Num_dofs[0]: %d, Num_chords[0]: %d\n", 
	 Num_elements[0], Num_dofs[0], Num_chords[0]);
  
  ierr = hypre_AMGeMatrixAssemble(&Matrix[0],

				  i_element_chord[0],
				  j_element_chord[0],
				  a_element_chord[0],

				  i_chord_dof[0], 
				  j_chord_dof[0],

				  Num_elements[0], 
				  Num_chords[0],
				  Num_dofs[0]);

  hypre_printf("nnz[0]: %d\n", hypre_CSRMatrixI(Matrix[0])[Num_dofs[0]]);
  /* impose Dirichlet boundary conditions: -----------------*/
  hypre_printf("imposing Dirichlet boundary conditions:====================\n");

  i_dof_dof_a = hypre_CSRMatrixI(Matrix[0]);
  j_dof_dof_a = hypre_CSRMatrixJ(Matrix[0]);
  a_dof_dof   = hypre_CSRMatrixData(Matrix[0]);
  for (i=0; i < Num_dofs[0]; i++)
    for (j=i_dof_dof_a[i]; j < i_dof_dof_a[i+1]; j++)
      if (i_dof_on_boundary[j_dof_dof_a[j]] == 0 
	  &&j_dof_dof_a[j]!=i)
	a_dof_dof[j] = 0.e0;

  for (i=0; i < Num_dofs[0]; i++)
    for (j=i_dof_dof_a[i]; j < i_dof_dof_a[i+1]; j++)
      if (i_dof_on_boundary[i] == 0 &&  j_dof_dof_a[j] !=i)
	a_dof_dof[j] = 0.e0;

  /*
  num_faces = hypre_AMGeMatrixTopologyNumFaces(A[0]);

  i_face_to_prefer_weight = hypre_CTAlloc(HYPRE_Int, num_faces);
  i_face_weight = hypre_CTAlloc(HYPRE_Int, num_faces);
  */


  /* build aggreagtes from elements: ----------------------------- */

  ierr = transpose_matrix_create(&i_dof_element,
				 &j_dof_element,

				 i_element_dof[0],
				 j_element_dof[0],

				 Num_elements[0], 
				 Num_dofs[0]);

  i_dof_aggregate = hypre_CTAlloc(HYPRE_Int, Num_dofs[0]+1);
  j_dof_aggregate = hypre_CTAlloc(HYPRE_Int, Num_dofs[0]);
  for (i=0; i < Num_dofs[0]; i++)
    {
      i_dof_aggregate[i] = i;
      j_dof_aggregate[i] = -1;
      for (j=i_dof_element[i]; j < i_dof_element[i+1]; j++)
	if (j_dof_aggregate[i] < j_dof_element[j])
	 j_dof_aggregate[i] =  j_dof_element[j];
    }

  i_dof_aggregate[Num_dofs[0]] = Num_dofs[0];

  ierr =
    transpose_matrix_create(&i_aggregate_dof,
			    &j_aggregate_dof,

			    i_dof_aggregate,
			    j_dof_aggregate,

			    Num_dofs[0], Num_elements[0]);


  hypre_TFree(i_dof_aggregate);
  hypre_TFree(j_dof_aggregate);
  hypre_TFree(i_dof_element);
  hypre_TFree(j_dof_element);

  level = 0;



interpolation_step:
  /*
  A[level+1] = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1); 
  hypre_CreateAMGeMatrixTopology(A[level+1]);

  ierr = hypre_CoarsenAMGeMatrixTopology(A[level+1],
					 A[level], 
				    
					 i_face_to_prefer_weight,
					 i_face_weight);


  */

  num_AEs = hypre_AMGeMatrixTopologyNumElements(A[level+1]);
  Num_elements[level+1] = num_AEs;
  hypre_printf("level %d num_AEs: %d\n\n\n", level+1, num_AEs);

  i_element_dof[level] = hypre_AMGeMatrixTopologyIElementNode(A[level]);
  j_element_dof[level] = hypre_AMGeMatrixTopologyJElementNode(A[level]);
  i_AE_element =  hypre_AMGeMatrixTopologyIAEElement(A[level+1]);
  j_AE_element =  hypre_AMGeMatrixTopologyJAEElement(A[level+1]);


  ierr = transpose_matrix_create(&i_dof_element,
				 &j_dof_element,

				 i_element_dof[level],
				 j_element_dof[level],

				 Num_elements[level], 
				 Num_dofs[level]);


  ierr = 
    hypre_AMGeSpectralBuildInterpolation(&P[level],

					 &i_coarseaggregate_coarsedof,
					 &j_coarseaggregate_coarsedof,

					 &i_element_dof[level+1],
					 &j_element_dof[level+1],

					 &i_element_chord[level+1],
					 &j_element_chord[level+1],
					 &a_element_chord[level+1],

					 &i_chord_dof[level+1],
					 &j_chord_dof[level+1],

					 &Num_chords[level+1],

					 &i_dof_coarsedof[level],
					 &j_dof_coarsedof[level],

					 &Num_dofs[level+1],
				      
					 i_element_dof[level],
					 j_element_dof[level],

					 i_dof_element, 
					 j_dof_element,

					 i_element_chord[level],
					 j_element_chord[level],
					 a_element_chord[level],


					 i_aggregate_dof,
					 j_aggregate_dof,


					 i_AE_element, j_AE_element,


					 i_chord_dof[level],
					 j_chord_dof[level],
				 
					 tolerance, 

					 Num_chords[level],

					 Num_elements[level+1],
					 
					 Num_elements[level], 
					 Num_dofs[level]);

  hypre_TFree(i_dof_element);
  hypre_TFree(j_dof_element);


  hypre_AMGeMatrixTopologyIElementNode(A[level+1]) = 
    i_element_dof[level+1];
  hypre_AMGeMatrixTopologyJElementNode(A[level+1]) = 
    j_element_dof[level+1];
  hypre_AMGeMatrixTopologyNumNodes(A[level+1]) = Num_dofs[level+1];

  hypre_printf("num_dofs[%d]: %d\n", level+1, Num_dofs[level+1]);


  /*
  for (i=0; i < Num_dofs[level]; i++)
    for (j=i_dof_coarsedof[level][i]; j<i_dof_coarsedof[level][i+1]; j++)
      hypre_printf("dof: %d, coarsedof: %d\n", i, j_dof_coarsedof[level][j]);
      */


  if (hypre_AMGeMatrixTopologyIFaceNode(A[level+1]))
    hypre_TFree(hypre_AMGeMatrixTopologyIFaceNode(A[level+1]));

  if (hypre_AMGeMatrixTopologyJFaceNode(A[level+1]))
    hypre_TFree(hypre_AMGeMatrixTopologyJFaceNode(A[level+1]));

  hypre_AMGeMatrixTopologyIFaceNode(A[level+1]) = NULL;
  hypre_AMGeMatrixTopologyJFaceNode(A[level+1]) = NULL;


  hypre_TFree(i_aggregate_dof);
  hypre_TFree(j_aggregate_dof);

  i_aggregate_dof = i_coarseaggregate_coarsedof;
  j_aggregate_dof = j_coarseaggregate_coarsedof;

  hypre_printf("END building level[%d] Interpolation: -------------------\n", level);

  hypre_printf("\nB U I L D I N G  level[%d]  S T I F F N E S S   M A T R I X\n", 
	 level+1);
  hypre_printf("\n  ==================         R A P       ===================\n");


  ierr = hypre_AMGeRAP(&Matrix[level+1], Matrix[level], P[level]);
  hypre_printf("END building coarse matrix; -----------------------------------\n");

  hypre_printf("nnz[%d]: %d\n", level+1, 
	 hypre_CSRMatrixI(Matrix[level+1])[Num_dofs[level+1]]);


  level++;
  if (num_AEs > 1 && level+1 < Max_level) 
    goto interpolation_step;


  hypre_printf("\n==============================================================\n");
  hypre_printf("number of grids: from 0 to %d\n\n\n", level);
  hypre_printf("==============================================================\n\n");


  /*
  hypre_TFree(i_face_to_prefer_weight);
  hypre_TFree(i_face_weight);

  */
  hypre_TFree(i_aggregate_dof);
  hypre_TFree(j_aggregate_dof);

  hypre_TFree(i_element_dof);
  hypre_TFree(j_element_dof);

  for (l=0; l < level; l++)
    {
      hypre_TFree(i_dof_coarsedof[l]);
      hypre_TFree(j_dof_coarsedof[l]);
    }

  hypre_TFree(i_dof_coarsedof);
  hypre_TFree(j_dof_coarsedof);


  for (l=1; l < level+1; l++)
    {
      hypre_TFree(i_element_chord[l]);
      hypre_TFree(j_element_chord[l]);
      hypre_TFree(a_element_chord[l]);

      hypre_TFree(i_chord_dof[l]);
      hypre_TFree(j_chord_dof[l]);
    }

  hypre_TFree(i_chord_dof);
  hypre_TFree(j_chord_dof);

  hypre_TFree(i_element_chord);
  hypre_TFree(j_element_chord);
  hypre_TFree(a_element_chord);

  *level_pointer = level;

  /*   *A_pointer = A;  */

  *P_pointer = P;
  *Matrix_pointer = Matrix;

  return ierr;

}
