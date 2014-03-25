#include <stdio.h>

HYPRE_Int 
matrix_matrix_product(    HYPRE_Int **i_element_edge_pointer, 
			  HYPRE_Int **j_element_edge_pointer,

			  HYPRE_Int *i_element_face, HYPRE_Int *j_element_face,
			  HYPRE_Int *i_face_edge, HYPRE_Int *j_face_edge,

			  HYPRE_Int num_elements, HYPRE_Int num_faces, HYPRE_Int num_edges)

{
  FILE *f;
  HYPRE_Int ierr =0, i, j, k, l, m;

  HYPRE_Int i_edge_on_local_list, i_edge_on_list;
  HYPRE_Int local_element_edge_counter = 0, element_edge_counter = 0;
  HYPRE_Int *j_local_element_edge;

  
  HYPRE_Int *i_element_edge, *j_element_edge;


  j_local_element_edge = (HYPRE_Int *) malloc((num_edges+1) * sizeof(HYPRE_Int));

  i_element_edge = (HYPRE_Int *) malloc((num_elements+1) * sizeof(HYPRE_Int));

  for (i=0; i < num_elements+1; i++)
    i_element_edge[i] = 0;

  for (i=0; i < num_elements; i++)
    {
      local_element_edge_counter = 0;
      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	{
	  k = j_element_face[j];

	  for (l=i_face_edge[k]; l < i_face_edge[k+1]; l++)
	    {
	      /* element i  and edge j_face_edge[l] are connected */
	    
	      /* hypre_printf("element %d  contains edge %d;\n",
		     i, j_face_edge[l]);  */

	      i_edge_on_local_list = -1;
	      for (m=0; m < local_element_edge_counter; m++)
		if (j_local_element_edge[m] == j_face_edge[l])
		  {
		    i_edge_on_local_list++;
		    break;
		  }

	      if (i_edge_on_local_list == -1)
		{
		  i_element_edge[i]++;
		  j_local_element_edge[local_element_edge_counter]=
		    j_face_edge[l];
		  local_element_edge_counter++;
		}
	    }
	}
    }

  free(j_local_element_edge);

  for (i=0; i < num_elements; i++)
    i_element_edge[i+1] += i_element_edge[i];

  for (i=num_elements; i>0; i--)
    i_element_edge[i] = i_element_edge[i-1];

  i_element_edge[0] = 0;

  j_element_edge = (HYPRE_Int *) malloc(i_element_edge[num_elements]
				     * sizeof(HYPRE_Int));

  /* fill--in the actual j_element_edge array: --------------------- */

  element_edge_counter = 0;
  for (i=0; i < num_elements; i++)
    {
      i_element_edge[i] = element_edge_counter;
      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	{
	  for (k=i_face_edge[j_element_face[j]];
	       k<i_face_edge[j_element_face[j]+1];k++)
	    {
	      /* check if edge j_face_edge[k] is already on list ***/

	      i_edge_on_list = -1;
	      for (l=i_element_edge[i];
		   l<element_edge_counter; l++)
		if (j_element_edge[l] == j_face_edge[k])
		  {
		    i_edge_on_list++;
		    break;
		  }

	      if (i_edge_on_list == -1) 
		{
		  if (element_edge_counter >= 
		      i_element_edge[num_elements])
		    {
		      hypre_printf("error in j_element_edge size: %d \n",
			     element_edge_counter);
		      break;
		    }

		  j_element_edge[element_edge_counter] =
		    j_face_edge[k];
		  element_edge_counter++;
		}
	    }
	}
		
    }

  i_element_edge[num_elements] = element_edge_counter;

  /*------------------------------------------------------------------
  f = fopen("element_edge", "w");
  for (i=0; i < num_elements; i++)
    {   
      hypre_printf("\nelement: %d has edges:\n", i);  
      for (j=i_element_edge[i]; j < i_element_edge[i+1]; j++)
	{
	  hypre_printf("%d ", j_element_edge[j]); 
	  hypre_fprintf(f, "%d %d\n", i, j_element_edge[j]);
	}
	  
      hypre_printf("\n"); 
    }

  fclose(f);
  */

  /* hypre_printf("end element_edge computation: ++++++++++++++++++++++++ \n");*/

  *i_element_edge_pointer = i_element_edge;
  *j_element_edge_pointer = j_element_edge;

  return ierr;

}
