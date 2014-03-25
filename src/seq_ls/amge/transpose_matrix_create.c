#include <stdio.h>

HYPRE_Int 
transpose_matrix_create(  HYPRE_Int **i_face_element_pointer,
			  HYPRE_Int **j_face_element_pointer,

			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,

			  HYPRE_Int num_elements, HYPRE_Int num_faces)

{
  FILE *f;
  HYPRE_Int ierr =0, i, j;

  HYPRE_Int *i_face_element, *j_face_element;

  /* ======================================================================
     first create face_element graph: -------------------------------------
     ====================================================================== */

  i_face_element = (HYPRE_Int *) malloc((num_faces+1) * sizeof(HYPRE_Int));
  j_face_element = (HYPRE_Int *) malloc(i_element_face[num_elements] * sizeof(HYPRE_Int));


  for (i=0; i < num_faces; i++)
    i_face_element[i] = 0;

  for (i=0; i < num_elements; i++)
    for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
      i_face_element[j_element_face[j]]++;

  i_face_element[num_faces] = i_element_face[num_elements];

  for (i=num_faces-1; i > -1; i--)
    i_face_element[i] = i_face_element[i+1] - i_face_element[i];

  for (i=0; i < num_elements; i++)
    for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
      {
	j_face_element[i_face_element[j_element_face[j]]] = i;
	i_face_element[j_element_face[j]]++;
      }

  for (i=num_faces-1; i > -1; i--)
    i_face_element[i+1] = i_face_element[i];

  i_face_element[0] = 0;

  /* hypre_printf("end building face--element graph: ++++++++++++++++++\n"); */

  /* END building face_element graph; ================================ */

  *i_face_element_pointer = i_face_element;
  *j_face_element_pointer = j_face_element;

  return ierr;

}
