/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/



#include "headers.h"  


/* unacceptable faces: i_face_to_prefer_weight[] = -1; ------------------*/


int hypre_AMGeAgglomerate(int *i_AE_element, int *j_AE_element,

			  int *i_face_face, int *j_face_face, int *w_face_face,
		     
			  int *i_face_element, int *j_face_element,
			  int *i_element_face, int *j_element_face,

			  int *i_face_to_prefer_weight,
			  int *i_face_weight,

			  int num_faces, int num_elements,
			  int *num_AEs_pointer)
{

  int ierr = 0;
  int i, j, k, l, m;

  int face_to_eliminate;
  int max_weight_old, max_weight;

  int AE_counter=0, AE_element_counter=0;

  int i_element_face_counter;

  int *i_element_to_AE;

  int *previous, *next, *first;
  int head, tail, last;

  int face_max_weight, face_local_max_weight, preferred_weight;

  int weight, weight_max;

  max_weight = 1;
  for (i=0; i < num_faces; i++)
    {
      weight = 1;
      for (j=i_face_face[i]; j < i_face_face[i+1]; j++)
	weight+= w_face_face[j];
      if (max_weight < weight) max_weight = weight;
    }

  first = hypre_CTAlloc(int, max_weight+1);



  next = hypre_CTAlloc(int, num_faces);


  previous = hypre_CTAlloc(int, num_faces+1);


  tail = num_faces;
  head = -1;

  for (i=0; i < num_faces; i++)
    {
      next[i] = i+1;
      previous[i] = i-1;
    }
  
  last = num_faces-1;
  previous[tail] = last;

  for (weight=1; weight <= max_weight; weight++)
    first[weight] = tail;

  i_element_to_AE = hypre_CTAlloc(int, num_elements);

  /*=======================================================================
                     AGGLOMERATION PROCEDURE:
    ======================================================================= */

  for (k=0; k < num_elements; k++) 
    i_element_to_AE[k] = -1;

  for (k=0; k < num_faces; k++) 
    i_face_weight[k] = 1;


  first[0] = 0;
  first[1] = 0;

  last = previous[tail];
  weight_max = i_face_weight[last];


  k = last;
  face_max_weight = -1;
  while (k!= head)
    {
      if (i_face_to_prefer_weight[k] > -1)
	face_max_weight = k;
	  
      if (face_max_weight > -1) break;
	  
      k=previous[k];
    }


  /* this will be used if the faces have been sorted: *****************
  k = last;
  face_max_weight = -1;
  while (k != head)
    {
      if (i_face_to_prefer_weight[k] > -1)
	face_max_weight = k;


      if (face_max_weight > -1) 
	{
	  max_weight = i_face_weight[face_max_weight];
	  l = face_max_weight;

	  while (previous[l] != head)
	    {

	      if (i_face_weight[previous[l]] < max_weight) 
		break;
	      else
		if (i_face_to_prefer_weight[previous[l]] > 
		    i_face_to_prefer_weight[face_max_weight])
		  {
		    l = previous[l];
		    face_max_weight = l;
		  }
		else
		  l = previous[l];
	    }

	  break; 
	}


      l =previous[k];



      weight = i_face_weight[k];
      last = previous[tail];
      if (last == head) 
	weight_max = 0;
      else
	weight_max = i_face_weight[last];


      ierr = remove_entry(weight, &weight_max, 
			  previous, next, first, &last,
			  head, tail, 
			  k);

			  



      k=l;
    }
    */

  if (face_max_weight == -1)
    {
      printf("all faces are unacceptable, i.e., no faces to eliminate !\n");

      *num_AEs_pointer = 1;

      i_AE_element[0] = 0;
      for (i=0; i < num_elements; i++)
	{
	  i_element_to_AE[i] = 0;
	  j_AE_element[i] = i;
	}

      i_AE_element[1] = num_elements;

      return ierr;
    }

  for (k=0; k < num_faces; k++)
    if (i_face_to_prefer_weight[k] > i_face_to_prefer_weight[face_max_weight])
      face_max_weight = k;

  max_weight = i_face_weight[face_max_weight];

  AE_counter=0;
  AE_element_counter=0;
   

  i_AE_element[AE_counter] = AE_element_counter;

  max_weight_old = -1;

  face_local_max_weight = face_max_weight;

eliminate_face:

  face_to_eliminate = face_local_max_weight;

  max_weight = i_face_weight[face_to_eliminate]; 

  last = previous[tail];
  if (last == head) 
    weight_max = 0;
  else
    weight_max = i_face_weight[last];

		   
  ierr = remove_entry(max_weight, &weight_max, 
		      previous, next, first, &last,
		      head, tail, 
		      face_to_eliminate);

  i_face_weight[face_to_eliminate] = 0;

  /*----------------------------------------------------------
   *  agglomeration step: 
   *
   *  put on AE_element -- list all elements 
   *  that share face "face_to_eliminate";
   *----------------------------------------------------------*/

  for (k = i_face_element[face_to_eliminate];
       k < i_face_element[face_to_eliminate+1]; k++)
    {
      /* check if element j_face_element[k] is already on the list: */

      if (j_face_element[k] < num_elements)
	{
	  if (i_element_to_AE[j_face_element[k]] == -1)
	    {
	      j_AE_element[AE_element_counter] = j_face_element[k];
	      i_element_to_AE[j_face_element[k]] = AE_counter;
	      AE_element_counter++;
	    }
	}
    }	  


  /* local update & search:==================================== */

  for (j=i_face_face[face_to_eliminate];
       j<i_face_face[face_to_eliminate+1]; j++)
    if (i_face_weight[j_face_face[j]] > 0)
      {
	weight = i_face_weight[j_face_face[j]];


	last = previous[tail];
	if (last == head) 
	  weight_max = 0;
	else
	  weight_max = i_face_weight[last];

	ierr = move_entry(weight, &weight_max, 
			  previous, next, first, &last,
			  head, tail, 
			  j_face_face[j]);

	i_face_weight[j_face_face[j]]+=w_face_face[j];

	weight = i_face_weight[j_face_face[j]];

	/* printf("update entry: %d\n", j_face_face[j]);   */

	last = previous[tail];
	if (last == head) 
	  weight_max = 0;
	else
	  weight_max = i_face_weight[last];

	
	ierr = update_entry(weight, &weight_max, 
			    previous, next, first, &last,
			    head, tail, 
			    j_face_face[j]);

	last = previous[tail];
	if (last == head) 
	  weight_max = 0;
	else
	  weight_max = i_face_weight[last];
		
      }

  /* find a face of the elements that have already been agglomerated
     with a maximal weight: ====================================== */
	  
  max_weight_old = max_weight;

  face_local_max_weight = -1; 
  preferred_weight = -1;

  for (l = i_AE_element[AE_counter];
       l < AE_element_counter; l++)
    {
      for (j=i_element_face[j_AE_element[l]];
	   j<i_element_face[j_AE_element[l]+1]; j++)
	{
	  i = j_element_face[j];

	  if (max_weight_old > 1 && i_face_weight[i] > 0 &&
	      i_face_to_prefer_weight[i] > -1)
	    {
	      if ( max_weight < i_face_weight[i])
		{
		  face_local_max_weight = i;
		  max_weight = i_face_weight[i];
		  preferred_weight = i_face_to_prefer_weight[i];
		}

	      if ( max_weight == i_face_weight[i]
		   && i_face_to_prefer_weight[i] > preferred_weight)
		{
		  face_local_max_weight = i;
		  preferred_weight = i_face_to_prefer_weight[i];
		}

	    }		
	}
    }  

  if (face_local_max_weight > -1) goto eliminate_face;

  /* ----------------------------------------------------------------
   * eliminate and label with i_face_weight[ ] = -1
   * "boundary faces of agglomerated elements";
   * those faces will be preferred for the next coarse spaces 
   * in case multiple coarse spaces are to be built;    
   * ---------------------------------------------------------------*/

  for (k = i_AE_element[AE_counter]; k < AE_element_counter; k++)
    {
      for (j = i_element_face[j_AE_element[k]];
	   j < i_element_face[j_AE_element[k]+1]; j++)
	{
	  if (i_face_weight[j_element_face[j]] > 0)
	    {
	      weight = i_face_weight[j_element_face[j]];
	      last = previous[tail];
	      if (last == head) 
		weight_max = 0;
	      else
		weight_max = i_face_weight[last];


	      ierr = remove_entry(weight, &weight_max, 
				  previous, next, first, &last,
				  head, tail, 
				  j_element_face[j]);

	      i_face_weight[j_element_face[j]] = -1;

	    }
	}
    }
      
  if (AE_element_counter > i_AE_element[AE_counter]) 
    {
      /* printf("completing agglomerated element: %d\n", 
		  AE_counter);   */ 
      AE_counter++;
    }

  i_AE_element[AE_counter] = AE_element_counter;
      

  /* find a face with maximal weight: ---------------------------*/


  last = previous[tail];
  if (last == head) goto end_agglomerate;

  weight_max = i_face_weight[last];

      
  /* printf("global search: ======================================\n"); */

  face_max_weight = -1;

  k = last;
  while (k != head)
    {
      if (i_face_to_prefer_weight[k] > -1)
	face_max_weight = k;


      if (face_max_weight > -1) 
	{
	  max_weight = i_face_weight[face_max_weight];
	  l = face_max_weight;

	  while (previous[l] != head)
	    {

	      if (i_face_weight[previous[l]] < max_weight) 
		break;
	      else
		if (i_face_to_prefer_weight[previous[l]] > 
		    i_face_to_prefer_weight[face_max_weight])
		  {
		    l = previous[l];
		    face_max_weight = l;
		  }
		else
		  l = previous[l];
	    }

	  break; 
	}


      l =previous[k];
      /* remove face k: ---------------------------------------*/


      weight = i_face_weight[k];
      last = previous[tail];
      if (last == head) 
	weight_max = 0;
      else
	weight_max = i_face_weight[last];


      ierr = remove_entry(weight, &weight_max, 
			  previous, next, first, &last,
			  head, tail, 
			  k);

			  
      /* i_face_weight[k] = -1; */


      k=l;
    }

  if (face_max_weight == -1) goto end_agglomerate;

  max_weight = i_face_weight[face_max_weight];

  face_local_max_weight = face_max_weight;

  goto eliminate_face;

end_agglomerate:


  /* eliminate isolated elements: ----------------------------------*/

  for (i=0; i<num_elements; i++)
    {

      if (i_element_to_AE[i] == -1)
	{
	  for (j=i_element_face[i]; j < i_element_face[i+1]
		 && i_element_to_AE[i] == -1; j++)
	    if (i_face_to_prefer_weight[j_element_face[j]] > -1)
	      for (k=i_face_element[j_element_face[j]];
		   k<i_face_element[j_element_face[j]+1]
		     && i_element_to_AE[i] == -1; k++)
		if (i_element_to_AE[j_face_element[k]] != -1)
		  i_element_to_AE[i] = i_element_to_AE[j_face_element[k]];
	}

      /*
      if (i_element_to_AE[i] == -1)
	{
	  i_element_face_counter = 0;
	  for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
	    if (i_face_to_prefer_weight[j_element_face[j]] > -1)
	      i_element_face_counter++;

	  if (i_element_face_counter == 1)
	    {
	      for (j=i_element_face[i]; j < i_element_face[i+1]; j++)
		if (i_face_to_prefer_weight[j_element_face[j]] > -1)
		  for (k=i_face_element[j_element_face[j]];
		       k<i_face_element[j_element_face[j]+1]; k++)
		    if (i_element_to_AE[j_face_element[k]] != -1)
		      i_element_to_AE[i] = i_element_to_AE[j_face_element[k]];
	    }
	}
	*/

      if (i_element_to_AE[i] == -1)
	{
	  i_element_to_AE[i] = AE_counter;
	  AE_counter++;
	}
    }
	  
  num_AEs_pointer[0] = AE_counter;


  /* compute adjoint graph: -------------------------------------------*/

  for (i=0; i < AE_counter; i++)
    i_AE_element[i] = 0;

  for (i=0; i < num_elements; i++)
    i_AE_element[i_element_to_AE[i]]++;

  i_AE_element[AE_counter] = num_elements;

  for (i=AE_counter-1; i > -1; i--)
    i_AE_element[i] = i_AE_element[i+1] - i_AE_element[i];

  for (i=0; i < num_elements; i++)
    {
      j_AE_element[i_AE_element[i_element_to_AE[i]]] = i;
      i_AE_element[i_element_to_AE[i]]++;
    }

  for (i=AE_counter-1; i > -1; i--)
    i_AE_element[i+1] = i_AE_element[i];

  i_AE_element[0] = 0;

  /*--------------------------------------------------------------------*/
  for (i=0; i < num_faces; i++)
    if (i_face_to_prefer_weight[i] == -1) i_face_weight[i] = -1;


  hypre_TFree(i_element_to_AE);

  hypre_TFree(previous);
  hypre_TFree(next);
  hypre_TFree(first);

  return ierr;
}

int update_entry(int weight, int *weight_max, 
		 int *previous, int *next, int *first, int *last,
		 int head, int tail, 
		 int i)

{
  int ierr = 0, weight0;

  /*
  printf("update_entry i: %d\n", i);
  printf("next[%d]: %d\n", i, next[i]);
  */

  if (previous[i] != head) next[previous[i]] = next[i];
  previous[next[i]] = previous[i];


  if (first[weight] == tail)
    {
      if (weight <= weight_max[0]) 
	{
	  printf("ERROR IN UPDATE_ENTRY: ===================\n");
	  printf("weight: %d, weight_max: %d\n",
		 weight, weight_max[0]);
	  return -1;
	}
      for (weight0=weight_max[0]+1; weight0 <= weight; weight0++)
	{
	  first[weight0] = i;
	  /* printf("create first[%d] = %d\n", weight0, i); */
	}

      /*
      printf("tail: %d, previous[tail]: %d\n", tail, previous[tail]);
      */
      previous[i] = previous[tail];
      next[i] = tail;
      if (previous[tail] > head) 
	next[previous[tail]] = i;
      previous[tail] = i;

    }
  else
    /* first[weight] already exists: =====================*/
    {
      previous[i] = previous[first[weight]];
      next[i] = first[weight];
      
      if (previous[first[weight]] != head)
	next[previous[first[weight]]] = i;

      previous[first[weight]] = i;

      for (weight0=1; weight0 <= weight; weight0++)
	if (first[weight0] == first[weight])
	  first[weight0] = i;

    }


  return ierr;
    
}

int remove_entry(int weight, int *weight_max, 
		 int *previous, int *next, int *first, int *last,
		 int head, int tail, 
		 int i)
{
  int ierr=0, weight0;

  if (previous[i] != head) next[previous[i]] = next[i];
  previous[next[i]] = previous[i];

  for (weight0=1; weight0 <= weight_max[0]; weight0++)
    {
      /* printf("first[%d}: %d\n", weight0,  first[weight0]); */
      if (first[weight0] == i)
	{
	  first[weight0] = next[i];
	  /* printf("shift: first[%d]= %d to %d\n",
		 weight0, i, next[i]);
	  if (i == last[0]) 
	    printf("i= last[0]: %d\n", i); */
	}
    }

  next[i] = i;
  previous[i] = i;

  return ierr;

}

int move_entry(int weight, int *weight_max, 
	       int *previous, int *next, int *first, int *last,
	       int head, int tail, 
	       int i)
{
  int ierr=0, weight0;

  if (previous[i] != head) next[previous[i]] = next[i];
  previous[next[i]] = previous[i];

  for (weight0=1; weight0 <= weight_max[0]; weight0++)
    {
      if (first[weight0] == i)
	first[weight0] = next[i];
    }

  return ierr;

}


