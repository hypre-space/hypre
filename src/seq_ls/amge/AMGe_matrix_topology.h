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





/*
#include <HYPRE_config.h>

#include "HYPRE_amge.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "_hypre_utilities.h"
#include "seq_mv.h"

*/


/*--------------------------------------------------------------------------
 * hypre_AMGeMatrixTopology:
 *--------------------------------------------------------------------------*/

typedef struct
{

  HYPRE_Int num_elements;
  HYPRE_Int num_faces;
  HYPRE_Int num_nodes;

  HYPRE_Int num_boundarysurfaces;

  HYPRE_Int *i_AE_element;
  HYPRE_Int *j_AE_element;

  HYPRE_Int *i_element_node;
  HYPRE_Int *j_element_node;

  HYPRE_Int *i_element_face;
  HYPRE_Int *j_element_face;

  HYPRE_Int *i_face_node;
  HYPRE_Int *j_face_node;

  HYPRE_Int *i_face_face;
  HYPRE_Int *j_face_face;
  
  HYPRE_Int *i_face_element;
  HYPRE_Int *j_face_element;

  HYPRE_Int *i_boundarysurface_face;
  HYPRE_Int *j_boundarysurface_face;


} hypre_AMGeMatrixTopology;

/*--------------------------------------------------------------------------
 * Accessor functions for the AMGe Matrix Topology structure
 *--------------------------------------------------------------------------*/

#define hypre_AMGeMatrixTopologyNumElements(matrix)   ((matrix) -> num_elements) 
#define hypre_AMGeMatrixTopologyNumFaces(matrix)      ((matrix) -> num_faces) 
#define hypre_AMGeMatrixTopologyNumNodes(matrix)     ((matrix) -> num_nodes) 
#define hypre_AMGeMatrixTopologyNumBoundarysurfaces(matrix)     ((matrix) -> num_boundarysurfaces) 


#define hypre_AMGeMatrixTopologyIFaceFace(matrix) ((matrix) -> i_face_face) 
#define hypre_AMGeMatrixTopologyJFaceFace(matrix) ((matrix) -> j_face_face) 

#define hypre_AMGeMatrixTopologyIAEElement(matrix) ((matrix) -> i_AE_element) 
#define hypre_AMGeMatrixTopologyJAEElement(matrix) ((matrix) -> j_AE_element) 



#define hypre_AMGeMatrixTopologyIElementNode(matrix) ((matrix) -> i_element_node) 
#define hypre_AMGeMatrixTopologyJElementNode(matrix) ((matrix) -> j_element_node) 

#define hypre_AMGeMatrixTopologyIElementFace(matrix)  ((matrix) -> i_element_face) 
#define hypre_AMGeMatrixTopologyJElementFace(matrix)  ((matrix) -> j_element_face) 

#define hypre_AMGeMatrixTopologyIFaceElement(matrix)  ((matrix) -> i_face_element) 
#define hypre_AMGeMatrixTopologyJFaceElement(matrix)  ((matrix) -> j_face_element) 


#define hypre_AMGeMatrixTopologyIFaceNode(matrix)    ((matrix) -> i_face_node) 
#define hypre_AMGeMatrixTopologyJFaceNode(matrix)    ((matrix) -> j_face_node) 


#define hypre_AMGeMatrixTopologyIBoundarysurfaceFace(matrix) ((matrix) -> i_boundarysurface_face)

#define hypre_AMGeMatrixTopologyJBoundarysurfaceFace(matrix) ((matrix) -> j_boundarysurface_face)

/*

#endif
*/
