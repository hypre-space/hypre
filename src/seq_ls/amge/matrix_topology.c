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

/* 
hypre_AMGeMatrixTopology * 
*/
HYPRE_Int
hypre_CreateAMGeMatrixTopology(hypre_AMGeMatrixTopology *matrix )

{
  

  /* matrix = hypre_CTAlloc(hypre_AMGeMatrixTopology, 1); */

  hypre_AMGeMatrixTopologyNumElements(matrix)   = 0;

  hypre_AMGeMatrixTopologyNumFaces(matrix)      = 0;
  hypre_AMGeMatrixTopologyNumNodes(matrix)     = 0;

  hypre_AMGeMatrixTopologyIFaceFace(matrix) = NULL;
  hypre_AMGeMatrixTopologyJFaceFace(matrix) = NULL;

  hypre_AMGeMatrixTopologyIAEElement(matrix)  = NULL;
  hypre_AMGeMatrixTopologyJAEElement(matrix)  = NULL;

  hypre_AMGeMatrixTopologyIElementNode(matrix) = NULL;
  hypre_AMGeMatrixTopologyJElementNode(matrix) = NULL;

  hypre_AMGeMatrixTopologyIElementFace(matrix)  = NULL;
  hypre_AMGeMatrixTopologyJElementFace(matrix)  = NULL;

  hypre_AMGeMatrixTopologyIFaceElement(matrix)  = NULL;
  hypre_AMGeMatrixTopologyJFaceElement(matrix)  = NULL;

  hypre_AMGeMatrixTopologyIFaceNode(matrix)    = NULL;
  hypre_AMGeMatrixTopologyJFaceNode(matrix)    = NULL;

  hypre_AMGeMatrixTopologyIBoundarysurfaceFace(matrix) = NULL;

  hypre_AMGeMatrixTopologyJBoundarysurfaceFace(matrix) = NULL;

  hypre_AMGeMatrixTopologyNumBoundarysurfaces(matrix) = 0;

  /* return matrix; */

  return 0;
}
/*--------------------------------------------------------------------------
 * hypre_DestroyAMGeMatrixToplogy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DestroyAMGeMatrixTopology( hypre_AMGeMatrixTopology *matrix )
{
   HYPRE_Int   ierr = 0;

   if (matrix)
   {
      if (hypre_AMGeMatrixTopologyIFaceFace(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyIFaceFace(matrix));
      if (hypre_AMGeMatrixTopologyJFaceFace(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyJFaceFace(matrix));

      if (hypre_AMGeMatrixTopologyIAEElement(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyIAEElement(matrix));
      if (hypre_AMGeMatrixTopologyJAEElement(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyJAEElement(matrix));

      if (hypre_AMGeMatrixTopologyIElementNode(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyIElementNode(matrix));
      if (hypre_AMGeMatrixTopologyJElementNode(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyJElementNode(matrix));

      if (hypre_AMGeMatrixTopologyIElementFace(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyIElementFace(matrix));
      if (hypre_AMGeMatrixTopologyJElementFace(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyJElementFace(matrix));

      if (hypre_AMGeMatrixTopologyIFaceElement(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyIFaceElement(matrix));
      if (hypre_AMGeMatrixTopologyJFaceElement(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyJFaceElement(matrix));


      if (hypre_AMGeMatrixTopologyIFaceNode(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyIFaceNode(matrix));
      if (hypre_AMGeMatrixTopologyJFaceNode(matrix))
         hypre_TFree(hypre_AMGeMatrixTopologyJFaceNode(matrix));

      if (hypre_AMGeMatrixTopologyNumBoundarysurfaces(matrix) >0)
	{
	  if (hypre_AMGeMatrixTopologyIBoundarysurfaceFace(matrix))
	    hypre_TFree(hypre_AMGeMatrixTopologyIBoundarysurfaceFace(matrix));

	  if (hypre_AMGeMatrixTopologyJBoundarysurfaceFace(matrix))
	    hypre_TFree(hypre_AMGeMatrixTopologyJBoundarysurfaceFace(matrix));
	}

      hypre_TFree(matrix);
   }

   return ierr;
}

