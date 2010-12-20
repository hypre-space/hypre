/*----------------------------------------------------------------------
 * intersect_elements.c
 *----------------------------------------------------------------------*/

/*=======================================================================
  creates intersecton sets e_1 \cap e_2, where
  e_1 and e_2 run over a e--e graph;


  input:    i_element_node, j_element_node;
            i_element_element, j_element_element;

            num_elements, num_nodes;


	    & boundary information:

	      i_boundary_surface_node, j_boundary_surface_node,
	      num_boundary_surfaces, 


  output:   i_face_node, j_face_node;
            i_face_element, j_face_element; HERE face_element contains 
	                                    boundary_surfaces as well;
					    their # = actual + num_elements;
            num_faces;


  temporary arrays: bit_vec of size num_nodes;
                    i_element_boundary_surface of size num_elements, 
		    j_element_boundary_surface of size <=
		              num_elements times num_boundary_surfaces;

  ====================================================================== */


HYPRE_Int create_maximal_intersection_sets(HYPRE_Int *i_element_node, HYPRE_Int *j_element_node,
				     HYPRE_Int *i_node_element, HYPRE_Int *j_node_element,
				     HYPRE_Int *i_element_element, 
				     HYPRE_Int *j_element_element,


				     HYPRE_Int *i_element_boundary_surface,
				     HYPRE_Int *j_element_boundary_surface,

				     HYPRE_Int *i_boundary_surface_node,  
				     HYPRE_Int *j_boundary_surface_node,
				     HYPRE_Int num_boundary_surfaces, 

				     HYPRE_Int num_elements, HYPRE_Int num_nodes,

				     HYPRE_Int **i_face_element_pointer, 
				     HYPRE_Int **j_face_element_pointer,

				     HYPRE_Int **i_face_node_pointer, 
				     HYPRE_Int **j_face_node_pointer,

				     HYPRE_Int *num_faces_pointer)

{
  HYPRE_Int ierr = 0;

  HYPRE_Int i,j,k,l,m,n;

  HYPRE_Int *bit_vec;

  HYPRE_Int num_local_faces = 0, num_local_boundary_faces = 0, 
    num_local_true_faces = 0;
  HYPRE_Int max_num_nodes_on_face = 0;

  HYPRE_Int face_counter = 0, face_node_counter = 0, face_element_counter = 0;

  HYPRE_Int local_face_counter, local_face_node_counter;


  HYPRE_Int *i_face_element, *j_face_element;
  HYPRE_Int *i_face_node, *j_face_node;

  HYPRE_Int *i_local_face_node, *j_local_face_node;
  HYPRE_Int *i_local_face_index;

  HYPRE_Int *local_element_num;




  for (i=0; i < num_elements; i++)
    {
      if (max_num_nodes_on_face < i_element_node[i+1]- i_element_node[i])
	max_num_nodes_on_face = i_element_node[i+1]- i_element_node[i];
      if (num_local_faces < i_element_element[i+1]- i_element_element[i])
	num_local_faces = i_element_element[i+1]- i_element_element[i];
    }

  num_local_boundary_faces = 0;
  if (num_boundary_surfaces !=0)
    for (i=0; i < num_elements; i++)
      {
	if (num_local_boundary_faces < i_element_boundary_surface[i+1]
                	  - i_element_boundary_surface[i])
	  num_local_boundary_faces = i_element_boundary_surface[i+1]
                	  - i_element_boundary_surface[i];
      }

  num_local_faces += num_local_boundary_faces;

  i_local_face_node = (HYPRE_Int *) malloc((num_local_faces+1) * sizeof(HYPRE_Int));

  j_local_face_node = (HYPRE_Int *) malloc(max_num_nodes_on_face *
				     num_local_faces * sizeof(HYPRE_Int));

  i_local_face_index = (HYPRE_Int *) malloc(num_local_faces * sizeof(HYPRE_Int));

  bit_vec = (HYPRE_Int *) malloc(num_nodes * sizeof(HYPRE_Int));

  local_element_num = (HYPRE_Int *) malloc(num_local_faces * sizeof(HYPRE_Int));


  for (i=0; i < num_nodes; i++)
    bit_vec[i] = 0;

  /* loop over elements and then over their neighbors with bigger index */

  for (i=0; i< num_elements; i++)
    {
      local_face_counter = 0;
      local_face_node_counter =0;
      for (k=0; k < num_local_faces; k++)
	local_element_num[k] = -1;

      for (j=i_element_element[i]; j < i_element_element[i+1]; j++)
	{

	  /* if (j_element_element[j] > i) */

	  if (j_element_element[j] != i) 
	    {
	      local_element_num[local_face_counter] = j_element_element[j];

	      i_local_face_node[local_face_counter] = local_face_node_counter;
	      local_face_counter++;

	      for (k=i_element_node[j_element_element[j]];
		   k<i_element_node[j_element_element[j]+1]; k++)
		bit_vec[j_element_node[k]]++;

	      for (k=i_element_node[i]; k < i_element_node[i+1]; k++)
		if (bit_vec[j_element_node[k]] !=0)
		  {
		    j_local_face_node[local_face_node_counter] =
		      j_element_node[k];
		    local_face_node_counter++;
		  }

	      /*--------------------------------------------------------
	      if (local_face_node_counter - 
		  i_local_face_node[local_face_counter-1] > 3) 
		{
		  hypre_printf("elements %d and %d are identical\n", 
			 i, j_element_element[j]);
		  for (k=i_element_node[i]; k < i_element_node[i+1]; k++)
		    hypre_printf("%d ", j_element_node[k]);

		  hypre_printf("\n");
		  for (k = i_element_node[j_element_element[j]]; 
		       k < i_element_node[j_element_element[j]+1]; k++)
		    hypre_printf("%d ", j_element_node[k]);

		  hypre_printf("\n");
		}
		-------------------------------------------------------*/

	      for (k=i_element_node[j_element_element[j]];
		   k<i_element_node[j_element_element[j]+1]; k++)
		bit_vec[j_element_node[k]]--;


	    }
	}

      /* loop over boundary surfaces: -------------------------*/
      if (num_boundary_surfaces !=0)
	{
      for (j=i_element_boundary_surface[i];
	   j<i_element_boundary_surface[i+1]; j++)
	{

	  local_element_num[local_face_counter] = 
	    num_elements + j_element_boundary_surface[j];

	  i_local_face_node[local_face_counter] = local_face_node_counter;
	  local_face_counter++;

	  for (k=i_boundary_surface_node[j_element_boundary_surface[j]];
	       k<i_boundary_surface_node[j_element_boundary_surface[j]+1];
	       k++)
	    bit_vec[j_boundary_surface_node[k]]++;

	  for (k=i_element_node[i]; k < i_element_node[i+1]; k++)
	    if (bit_vec[j_element_node[k]] !=0)
	      {
		j_local_face_node[local_face_node_counter] =
		  j_element_node[k];
		local_face_node_counter++;
	      }

	  for (k=i_boundary_surface_node[j_element_boundary_surface[j]];
	       k<i_boundary_surface_node[j_element_boundary_surface[j]+1];
	       k++)
	    bit_vec[j_boundary_surface_node[k]]--;

	}
	
	}    

      /*--------------------------------------------------------------------
      hypre_printf("element %d: %d local intersection sets found\n", i,
	     local_face_counter);
        --------------------------------------------------------------------*/

      i_local_face_node[local_face_counter] = local_face_node_counter;
      
      /* local faces ready: ======================================== */

      /* find maximal intersection sets: =========================== */

      /* compare face_k and face_l: -------------------------------- 
	 if face_k \subset face_l (incl. face_k == face_l)
	         delete face_k, i.e., set i_local_face_index[k]++;
	 ----------------------------------------------------------- */

      num_local_true_faces = 0;
      for (k=0; k < local_face_counter; k++)
	{
	  i_local_face_index[k] = -1;
	  for (l=0; l < local_face_counter; l++)
	    {
	      if (l != k)
		{
		  for (m=i_local_face_node[l]; m < i_local_face_node[l+1]; m++)
		    bit_vec[j_local_face_node[m]] = 1;

		  for (m=i_local_face_node[k]; m < i_local_face_node[k+1]; m++)
		    if (bit_vec[j_local_face_node[m]] != 1) goto e_new;
	      	      
		  i_local_face_index[k]++;

		e_new:
		  for (m=i_local_face_node[l]; m < i_local_face_node[l+1]; m++)
		    bit_vec[j_local_face_node[m]]=0;
		}

	      if  (i_local_face_index[k] == 0) break;

	    }

	  /* put local true faces on the global list of faces: ========== */

	  if (i_local_face_index[k] == -1 && local_element_num[k] > i) 
	    /*     if (i_local_face_index[k] == -1)   */
	    {
	      /*-------------------------------------------------------------
	      hypre_printf("put local true face %d on the global list of faces:\n",
		     face_counter);
		------------------------------------------------------------*/

	      num_local_true_faces++;

	      /*
	      i_face_element[face_counter] = face_element_counter; 
	      i_face_node[face_counter] = face_node_counter;
	      */

	      face_counter++;

	      /* j_face_element[face_element_counter] = i; */
	      face_element_counter++;

	      if (local_element_num[k] < num_elements)
		{
		  face_element_counter++;
		  /* 
		     j_face_element[face_element_counter] = 
		     local_element_num[k];

		     */
		}

	      /*-------------------------------------------------------------
	      if (i_local_face_node[k+1] - i_local_face_node[k] != 3)
		hypre_printf("a face has num of nodes: %d\n", 
		       i_local_face_node[k+1] - i_local_face_node[k]);
	        ------------------------------------------------------------*/
	      for (m=i_local_face_node[k]; m < i_local_face_node[k+1]; m++)
		{
		  /* j_face_node[face_node_counter] = j_local_face_node[m]; */
		  face_node_counter++;
		}
	    }
	}

      /*---------------------------------------------------------------------
      hypre_printf("num_faces found for element %d: %d\n", i, num_local_true_faces);
        ---------------------------------------------------------------------*/
	     
    }

  /* hypre_printf("=============== face_counter:%d ============\n", face_counter); */


  i_face_node = (HYPRE_Int *) malloc((face_counter+1) * sizeof(HYPRE_Int));
  j_face_node = (HYPRE_Int *) malloc(face_node_counter * sizeof(HYPRE_Int));

  i_face_element = (HYPRE_Int *) malloc((face_counter+1) * sizeof(HYPRE_Int));
  j_face_element = (HYPRE_Int *) malloc(face_element_counter * sizeof(HYPRE_Int));
  

  face_counter = 0; 
  face_node_counter = 0;
  face_element_counter = 0;

  for (i=0; i< num_elements; i++)
    {
      local_face_counter = 0;
      local_face_node_counter =0;
      for (k=0; k < num_local_faces; k++)
	local_element_num[k] = -1;

      for (j=i_element_element[i]; j < i_element_element[i+1]; j++)
	{

	  /* if (j_element_element[j] > i) */

	  if (j_element_element[j] != i) 
	    {
	      local_element_num[local_face_counter] = j_element_element[j];

	      i_local_face_node[local_face_counter] = local_face_node_counter;
	      local_face_counter++;

	      for (k=i_element_node[j_element_element[j]];
		   k<i_element_node[j_element_element[j]+1]; k++)
		bit_vec[j_element_node[k]]++;

	      for (k=i_element_node[i]; k < i_element_node[i+1]; k++)
		if (bit_vec[j_element_node[k]] !=0)
		  {
		    j_local_face_node[local_face_node_counter] =
		      j_element_node[k];
		    local_face_node_counter++;
		  }
	      /* ----------------------------------------------------
	      if (local_face_node_counter - 
		  i_local_face_node[local_face_counter-1] > 3) 
		{
		  hypre_printf("elements %d and %d are identical\n", 
			 i, j_element_element[j]);
		  for (k=i_element_node[i]; k < i_element_node[i+1]; k++)
		    hypre_printf("%d ", j_element_node[k]);

		  hypre_printf("\n");
		  for (k = i_element_node[j_element_element[j]]; 
		       k < i_element_node[j_element_element[j]+1]; k++)
		    hypre_printf("%d ", j_element_node[k]);

		  hypre_printf("\n");
		}
		--------------------------------------------------------*/

	      for (k=i_element_node[j_element_element[j]];
		   k<i_element_node[j_element_element[j]+1]; k++)
		bit_vec[j_element_node[k]]--;


	    }
	}

      /* loop over boundary surfaces: -------------------------*/
      if (num_boundary_surfaces !=0)
	{
      
      for (j=i_element_boundary_surface[i];
	   j<i_element_boundary_surface[i+1]; j++)
	{

	  local_element_num[local_face_counter] = 
	    num_elements + j_element_boundary_surface[j];

	  i_local_face_node[local_face_counter] = local_face_node_counter;
	  local_face_counter++;

	  for (k=i_boundary_surface_node[j_element_boundary_surface[j]];
	       k<i_boundary_surface_node[j_element_boundary_surface[j]+1];
	       k++)
	    bit_vec[j_boundary_surface_node[k]]++;

	  for (k=i_element_node[i]; k < i_element_node[i+1]; k++)
	    if (bit_vec[j_element_node[k]] !=0)
	      {
		j_local_face_node[local_face_node_counter] =
		  j_element_node[k];
		local_face_node_counter++;
	      }

	  for (k=i_boundary_surface_node[j_element_boundary_surface[j]];
	       k<i_boundary_surface_node[j_element_boundary_surface[j]+1];
	       k++)
	    bit_vec[j_boundary_surface_node[k]]--;

	}
	
	}

      /*--------------------------------------------------------------------
      hypre_printf("element %d: %d local intersection sets found\n", i,
	     local_face_counter);
        --------------------------------------------------------------------*/

      i_local_face_node[local_face_counter] = local_face_node_counter;
      
      /* local faces ready: ======================================== */

      /* find maximal intersection sets: =========================== */

      /* compare face_k and face_l: -------------------------------- 
	 if face_k \subset face_l (incl. face_k == face_l)
	         delete face_k, i.e., set i_local_face_index[k]++;
	 ----------------------------------------------------------- */

      num_local_true_faces = 0;
      for (k=0; k < local_face_counter; k++)
	{
	  i_local_face_index[k] = -1;
	  for (l=0; l < local_face_counter; l++)
	    {
	      if (l != k)
		{
		  for (m=i_local_face_node[l]; m < i_local_face_node[l+1]; m++)
		    bit_vec[j_local_face_node[m]] = 1;

		  for (m=i_local_face_node[k]; m < i_local_face_node[k+1]; m++)
		    if (bit_vec[j_local_face_node[m]] != 1) goto e1_new;
	      	      
		  i_local_face_index[k]++;

		e1_new:
		  for (m=i_local_face_node[l]; m < i_local_face_node[l+1]; m++)
		    bit_vec[j_local_face_node[m]]=0;
		}

	      if  (i_local_face_index[k] == 0) break;

	    }

	  /* put local true faces on the global list of faces: ========== */

	  if (i_local_face_index[k] == -1 && local_element_num[k] > i) 
	    /*     if (i_local_face_index[k] == -1)   */
	    {
	      /*-------------------------------------------------------------
	      hypre_printf("put local true face %d on the global list of faces:\n",
		     face_counter);
		------------------------------------------------------------*/

	      num_local_true_faces++;

	      i_face_element[face_counter] = face_element_counter; 
	      i_face_node[face_counter] = face_node_counter;
	      face_counter++;

	      j_face_element[face_element_counter] = i;

	      face_element_counter++;

	      if (local_element_num[k] < num_elements)
		{
		  j_face_element[face_element_counter] = local_element_num[k];
		  face_element_counter++;
		}

	      /* ---------------------------------------------------------
	      if (i_local_face_node[k+1] - i_local_face_node[k] != 3)
		hypre_printf("a face has num of nodes: %d\n", 
		       i_local_face_node[k+1] - i_local_face_node[k]);

	       ---------------------------------------------------------*/
	      for (m=i_local_face_node[k]; m < i_local_face_node[k+1]; m++)
		{
		  j_face_node[face_node_counter] = j_local_face_node[m];
		  face_node_counter++;
		}
	    }
	}

      /*---------------------------------------------------------------------
      hypre_printf("num_faces found for element %d: %d\n", i, num_local_true_faces);
        ---------------------------------------------------------------------*/
	     
    }

  /* hypre_printf("=============== face_counter:%d ============\n", face_counter); */


  num_faces_pointer[0] = face_counter;

  i_face_node[face_counter] = face_node_counter;
  i_face_element[face_counter] = face_element_counter;

  free(bit_vec);

  free(i_local_face_node);
  free(j_local_face_node);
  free(i_local_face_index);
  free(local_element_num);

  *i_face_node_pointer = i_face_node;
  *j_face_node_pointer = j_face_node;

  *i_face_element_pointer = i_face_element;
  *j_face_element_pointer = j_face_element;

}
