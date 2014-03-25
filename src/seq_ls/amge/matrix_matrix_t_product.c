#include <stdio.h>

HYPRE_Int 
matrix_matrix_t_product(  HYPRE_Int **i_element_element_pointer, 
			  HYPRE_Int **j_element_element_pointer,

			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,

			  HYPRE_Int num_elements, HYPRE_Int num_faces)

{
  FILE *f;
  HYPRE_Int ierr =0, i, j, k, l, m;

  HYPRE_Int i_element_on_local_list, i_element_on_list;
  HYPRE_Int local_element_element_counter = 0, element_element_counter = 0;
  HYPRE_Int *j_local_element_element;

  
  HYPRE_Int *i_element_element, *j_element_element;
  HYPRE_Int *i_face_element,    *j_face_element;


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


  /* hypre_printf("============= create element_element graph=============\n"); */

  /* hypre_printf("by multiplying element_face and face_element graphs; ==\n"); */


  j_local_element_element = (HYPRE_Int *) malloc((num_elements+1) * sizeof(HYPRE_Int));

  i_element_element = (HYPRE_Int *) malloc((num_elements+1) * sizeof(HYPRE_Int));

  for (i=0; i < num_elements+1; i++)
    i_element_element[i] = 0;

  for (i=0; i < num_elements; i++)
    {
      local_element_element_counter = 0;
      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	{
	  k = j_element_face[j];

	  for (l=i_face_element[k]; l < i_face_element[k+1]; l++)
	    {
	      /* element i  and element j_face_element[l] are connected */
	    
	      /* hypre_printf("element %d  and element %d are connected;\n",
		     i, j_face_element[l]); */

	      i_element_on_local_list = -1;
	      for (m=0; m < local_element_element_counter; m++)
		if (j_local_element_element[m] == j_face_element[l])
		  {
		    i_element_on_local_list++;
		    break;
		  }

	      if (i_element_on_local_list == -1)
		{
		  i_element_element[i]++;
		  j_local_element_element[local_element_element_counter]=
		    j_face_element[l];
		  local_element_element_counter++;
		}
	    }
	}
    }


  for (i=0; i < num_elements; i++)
    i_element_element[i+1] += i_element_element[i];

  for (i=num_elements; i>0; i--)
    i_element_element[i] = i_element_element[i-1];

  i_element_element[0] = 0;

  j_element_element = (HYPRE_Int *) malloc(i_element_element[num_elements]
				     * sizeof(HYPRE_Int));

  /* fill--in the actual j_element_element array: --------------------- */

  element_element_counter = 0;
  for (i=0; i < num_elements; i++)
    {
      i_element_element[i] = element_element_counter;
      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	{
	  for (k=i_face_element[j_element_face[j]];
	       k<i_face_element[j_element_face[j]+1];k++)
	    {
	      /* check if element j_face_element[k] is already on list ***/
	      i_element_on_list = -1;
	      for (l=i_element_element[i];
		   l<element_element_counter; l++)
		if (j_element_element[l] == j_face_element[k])
		  {
		    i_element_on_list++;
		    break;
		  }
	      if (i_element_on_list == -1) 
		{
		  if (element_element_counter >= 
		      i_element_element[num_elements])
		    {
		      hypre_printf("error in j_elemenet_element size: %d \n",
			     element_element_counter);
		      break;
		    }

		  j_element_element[element_element_counter] =
		    j_face_element[k];
		  element_element_counter++;
		}
	    }
	}
		
    }


 i_element_element[num_elements] = element_element_counter;


  *i_element_element_pointer = i_element_element;
  *j_element_element_pointer = j_element_element;

  free(i_face_element);
  free(j_face_element);

  free(j_local_element_element);

  /*======================================================================
  f = fopen("element_element", "w");
  for (i=0; i < num_elements; i++)
    {   
      hypre_printf("\nelement: %d has elements:\n", i);  
      for (j=i_element_element[i]; j < i_element_element[i+1]; j++)
	{
	  hypre_printf("%d ", j_element_element[j]);
	  hypre_fprintf(f, "%d %d\n", i, j_element_element[j]);
	}
	  
      hypre_printf("\n"); 
    }

  fclose(f);



  hypre_printf("end element_element computation: ++++++++++++++++++++++++ \n"); 
  ====================================================================== */

  return ierr;

}
