
/*
#include <HYPRE_config.h>

#include "HYPRE_amge.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "utilities.h"
#include "seq_mv.h"

*/

/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * hypre_AMGeMatrixTopology:
 *--------------------------------------------------------------------------*/

typedef struct
{

  int num_elements;
  int num_faces;
  int num_nodes;

  int num_boundarysurfaces;

  int *i_AE_element;
  int *j_AE_element;

  int *i_element_node;
  int *j_element_node;

  int *i_element_face;
  int *j_element_face;

  int *i_face_node;
  int *j_face_node;

  int *i_face_face;
  int *j_face_face;
  
  int *i_face_element;
  int *j_face_element;

  int *i_boundarysurface_face;
  int *j_boundarysurface_face;


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
