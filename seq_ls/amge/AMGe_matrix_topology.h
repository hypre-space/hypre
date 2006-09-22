/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/*
#include <HYPRE_config.h>

#include "HYPRE_amge.h"

#ifndef hypre_LS_HEADER
#define hypre_LS_HEADER

#include "utilities.h"
#include "seq_mv.h"

*/


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
