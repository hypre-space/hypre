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




/*****************************************************************************
 * reads from file x_coord and y_coord;
 *                 and visualizes in 2d agglomerated elements;
 ****************************************************************************/

#include "headers.h"  


HYPRE_Int hypre_AMGe2dGraphics(hypre_AMGeMatrixTopology **A,
			 HYPRE_Int level,

			 HYPRE_Int *i_element_node_0, 
			 HYPRE_Int *j_element_node_0,

			 HYPRE_Int *Num_elements,

			 char *coordinates)

{

  FILE *f;
  HYPRE_Int ierr = 0;
  
  HYPRE_Int i,j,k,l,m;

  HYPRE_Int num_nodes, num_faces;

  HYPRE_Int *i_element_face, *j_element_face;
  HYPRE_Int *i_AE_element, *j_AE_element;
  HYPRE_Int *i_element_element, *j_element_element;
  
  /* visualization arrays: -------------------------------------------------*/
  float *red, *blue, *green;
  HYPRE_Int num_colors, i_color, max_color, facesize, no_color;
  float *x1, *y1;

  HYPRE_Int i_color_to_choose;

  HYPRE_Real *x_coord, *y_coord;

  HYPRE_Int *i_AE_color;
  HYPRE_Int *i_element_element_0 = hypre_AMGeMatrixTopologyIAEElement(A[0]);
  HYPRE_Int *j_element_element_0 = hypre_AMGeMatrixTopologyJAEElement(A[0]);

  HYPRE_Int num_elements_0;
  HYPRE_Int *i_AE_element_0, *j_AE_element_0;

  /*------------------------------------------------------------------------
   * initializing
   *------------------------------------------------------------------------*/

  red = hypre_CTAlloc(float, 256);
  blue = hypre_CTAlloc(float, 256);
  green = hypre_CTAlloc(float, 256);

  red[0] =  1;
  blue[0] = 0;
  green[0] = 0;

  red[1] =  0.;
  blue [1] = 1;
  green[1] = 0;

  red[2] =  0;
  blue [2] = 0;
  green[2] = 1;

  red[3] =  0;
  blue [3] = 1;
  green[3] = 1;


  red[4] =  1;
  blue [4] =0;    
  green[4] =1 ;

  red[5] =  1;
  blue [5] =1;    
  green[5] =0;

  red[6] =   0.1;
  blue [6] = 0.1;    
  green[6] = 0.1;

  red[7] =  0.1;
  blue [7] =1;    
  green[7] = 0;

  red[8] =  0.1;
  blue [8] =0.3;    
  green[8] = 1;

  red[9] =  1;
  blue [9] =0;    
  green[9] = 1;

  red[10] =  0.5;
  blue [10] =0;    
  green[10] = 1;

  red[11] =  0;
  blue [11] = 0;
  green[11] = 0;

  max_color = 11;

  no_color = -1;

  num_nodes = hypre_AMGeMatrixTopologyNumNodes(A[0]);
  Num_elements[0] = hypre_AMGeMatrixTopologyNumElements(A[0]);


  x1 = hypre_CTAlloc(float, num_nodes);
  y1 = hypre_CTAlloc(float, num_nodes);

  x_coord = hypre_CTAlloc(HYPRE_Real, num_nodes);
  y_coord = hypre_CTAlloc(HYPRE_Real, num_nodes);

  f = fopen(coordinates, "r");
  for( i = 0; i < num_nodes; i++ )
    hypre_fscanf(f, "%le %le\n", &x_coord[i], &y_coord[i]);


  fclose(f);


  i_AE_color = hypre_CTAlloc(HYPRE_Int, Num_elements[0]+1);


  for (l=0; l < level+1; l++)
    {

      Num_elements[l] = hypre_AMGeMatrixTopologyNumElements(A[l]);
      /* hypre_printf("level %d, num_elements: %d\n", l, Num_elements[l]); */


      i_element_face = hypre_AMGeMatrixTopologyIElementFace(A[l]);
      j_element_face = hypre_AMGeMatrixTopologyJElementFace(A[l]);
      num_faces = hypre_AMGeMatrixTopologyNumFaces(A[l]);

      ierr = matrix_matrix_t_product(&i_element_element, &j_element_element,

				     i_element_face, j_element_face,

				     Num_elements[l], num_faces);

 
      num_colors = 10;

    e0:
      for (i=0; i < Num_elements[l]; i++)
	i_AE_color[i] = -1;

      for (i=0; i < Num_elements[l]; i++)
	{
	  for (i_color = 0; i_color < num_colors; i_color++)
	    {
	      i_color_to_choose = -1;
	      for (j=i_element_element[i]; j < i_element_element[i+1]; j++)
		if (i_AE_color[j_element_element[j]] == i_color)
		  {
		    i_color_to_choose = 0;
		    break;
		  }
	      if (i_color_to_choose == -1)
		{
		  i_AE_color[i] = i_color;

		  break;
		}
	    }
	  if (i_AE_color[i] == -1) 
	    {
	      num_colors++;
	      goto e0;
	    }
	}


      hypre_TFree(i_element_element);
      hypre_TFree(j_element_element);

      xutl0_(&num_colors, red, green, blue );  

      i_AE_element = hypre_AMGeMatrixTopologyIAEElement(A[l]);
      j_AE_element = hypre_AMGeMatrixTopologyJAEElement(A[l]);

      if (l==0)
	{
	  i_AE_element_0 = i_AE_element;
	  j_AE_element_0 = i_AE_element;
	}
      else
	{
	  ierr = matrix_matrix_product(&i_AE_element_0,
				       &j_AE_element_0,

				       i_AE_element, j_AE_element,
			       
				       i_element_element_0,
				       j_element_element_0,
			       
				       Num_elements[l], Num_elements[l-1], 
				       Num_elements[0]);
	}
	  



      for (i=0; i < Num_elements[l]; i++)
	{
	  i_color = i_AE_color[i]+1;
	  if (i_color > 10) i_color = 1;
	  for (m=i_AE_element_0[i]; m < i_AE_element_0[i+1]; m++)
	    {
	      k = j_AE_element_0[m];

	      facesize=0;
	      for (j=i_element_node_0[k]; j < i_element_node_0[k+1]; j++)
		{
		  x1[facesize] = x_coord[j_element_node_0[j]];
		  y1[facesize] = y_coord[j_element_node_0[j]];

		  facesize++;
		}

	      xfill0_(x1, y1, &facesize, &i_color); 
	      x1[facesize] = x1[0];
	      y1[facesize] = y1[0];
	      facesize++;
	      xline0_(x1, y1, &facesize, &max_color);
	    } 
	}

      xutl0_(&no_color, red, green, blue );  

      if (l>1)
	{
	  hypre_TFree(i_element_element_0);
	  hypre_TFree(j_element_element_0);
	}


      i_element_element_0 = i_AE_element_0;
      j_element_element_0 = j_AE_element_0;

    }

  hypre_printf("end displaying: ====================================\n\n");


  hypre_TFree(i_AE_color);

  hypre_TFree(i_AE_element_0);
  hypre_TFree(j_AE_element_0);

  hypre_TFree(red);
  hypre_TFree(blue);
  hypre_TFree(green);

  hypre_TFree(x1);
  hypre_TFree(y1);

  hypre_TFree(x_coord);
  hypre_TFree(y_coord);

  return ierr;


}

