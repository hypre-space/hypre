#include <stdio.h>

int 
matrix_matrix_t_product(  int **i_element_element_pointer, 
			  int **j_element_element_pointer,

			  int *i_element_face, int *j_element_face,

			  int num_elements, int num_faces)

{
  FILE *f;
  int ierr =0, i, j, k, l, m;

  int i_element_on_local_list, i_element_on_list;
  int local_element_element_counter = 0, element_element_counter = 0;
  int *j_local_element_element;

  
  int *i_element_element, *j_element_element;
  int *i_face_element,    *j_face_element;


  /* ======================================================================
     first create face_element graph: -------------------------------------
     ====================================================================== */

  i_face_element = (int *) malloc((num_faces+1) * sizeof(int));
  j_face_element = (int *) malloc(i_element_face[num_elements] * sizeof(int));


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

  /* printf("end building face--element graph: ++++++++++++++++++\n"); */

  /* END building face_element graph; ================================ */


  /* printf("============= create element_element graph=============\n"); */

  /* printf("by multiplying element_face and face_element graphs; ==\n"); */


  j_local_element_element = (int *) malloc((num_elements+1) * sizeof(int));

  i_element_element = (int *) malloc((num_elements+1) * sizeof(int));

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
	    
	      /* printf("element %d  and element %d are connected;\n",
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

  j_element_element = (int *) malloc(i_element_element[num_elements]
				     * sizeof(int));

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
		      printf("error in j_elemenet_element size: %d \n",
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
      printf("\nelement: %d has elements:\n", i);  
      for (j=i_element_element[i]; j < i_element_element[i+1]; j++)
	{
	  printf("%d ", j_element_element[j]);
	  fprintf(f, "%d %d\n", i, j_element_element[j]);
	}
	  
      printf("\n"); 
    }

  fclose(f);



  printf("end element_element computation: ++++++++++++++++++++++++ \n"); 
  ====================================================================== */

  return ierr;

}
