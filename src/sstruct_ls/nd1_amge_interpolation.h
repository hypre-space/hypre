/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





#ifndef hypre_ND1_AMGE_INTERPOLATION
#define hypre_ND1_AMGE_INTERPOLATION

/*
  Function:  hypre_ND1AMGeInterpolation

  Defines an operator-dependent (AMGe) interpolation for the fine interior
  edges, given the (e.g. geometric) interpolation for the fine edges on the
  boundaries of coarse elements. The parameters are:

  Aee  [input]
  The stiffness matrix for lowest order Nedelec elements on the fine level.

  ELEM_iedge, FACE_iedge, EDGE_iedge  [input]
  coarse grid elements, faces and edges.

  ELEM_FACE, ELEM_EDGE  [input]

  edge_EDGE  [input/output]
  The interpolation from coarse to fine edges. This is a partially filled
  matrix, with set (and fixed) nonzero pattern. We assume that the rows
  corresponding to fine edges on the boundary of a coarse element are
  given and complete the construction by computing the rest of the entries.

  Note: If FACE_iedge == EDGE_iedge the input should describe a 2D problem.
*/
HYPRE_Int hypre_ND1AMGeInterpolation (hypre_ParCSRMatrix * Aee,
                                hypre_ParCSRMatrix * ELEM_iedge,
                                hypre_ParCSRMatrix * FACE_iedge,
                                hypre_ParCSRMatrix * EDGE_iedge,
                                hypre_ParCSRMatrix * ELEM_FACE,
                                hypre_ParCSRMatrix * ELEM_EDGE,
                                HYPRE_Int            num_OffProcRows,
                                hypre_MaxwellOffProcRow ** OffProcRows,
                                hypre_IJMatrix     * edge_EDGE);

/*
  Function: hypre_HarmonicExtension

  Defines the interpolation operator Pi:DOF->idof by harmonically extending
  Pb:DOF->bdof based on the operator A. Specifically,
                A = [Aii,Aib] is idof x (idof+bdof)
                P = [-Pi;Pb]  is (idof+bdof) x DOF
  and the function computes
                     Pi = Aii^{-1} Aib Pb.
  The columns in A and P use global numbering, while the rows are numbered
  according to the arrays idof and bdof. The only output parameter is Pi.
*/
HYPRE_Int hypre_HarmonicExtension (hypre_CSRMatrix *A,
                             hypre_CSRMatrix *P,
                             HYPRE_Int num_DOF, HYPRE_Int *DOF,
                             HYPRE_Int num_idof, HYPRE_Int *idof,
                             HYPRE_Int num_bdof, HYPRE_Int *bdof);

#endif
