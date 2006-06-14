/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

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
  Boolean matrices describing the fine edges which are interior for the
  coarse grid elements, faces and edges.

  ELEM_FACE, ELEM_EDGE  [input]
  Boolean matrices describing the coarse grid topology.

  edge_EDGE  [input/output]
  The interpolation from coarse to fine edges. This is a partially filled
  matrix, with set (and fixed) nonzero pattern. We assume that the rows
  corresponding to fine edges on the boundary of a coarse element are
  given and complete the construction by computing the rest of the entries.

  Note: If FACE_iedge == EDGE_iedge the input should describe a 2D problem.
*/
int hypre_ND1AMGeInterpolation (hypre_ParCSRMatrix * Aee,
                                hypre_ParCSRMatrix * ELEM_iedge,
                                hypre_ParCSRMatrix * FACE_iedge,
                                hypre_ParCSRMatrix * EDGE_iedge,
                                hypre_ParCSRMatrix * ELEM_FACE,
                                hypre_ParCSRMatrix * ELEM_EDGE,
                                int                  num_OffProcRows,
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
int hypre_HarmonicExtension (hypre_CSRMatrix *A,
                             hypre_CSRMatrix *P,
                             int num_DOF, int *DOF,
                             int num_idof, int *idof,
                             int num_bdof, int *bdof);

#endif
