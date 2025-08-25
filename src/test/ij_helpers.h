/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

HYPRE_ParCSRMatrix GenerateLaplacian( MPI_Comm comm,
                                      HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                                      HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                      HYPRE_Int ip, HYPRE_Int iq, HYPRE_Int ir,
                                      HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateSysLaplacian( MPI_Comm comm,
                                         HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                                         HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                         HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                         HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef( MPI_Comm comm,
                                              HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                                              HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                              HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                              HYPRE_Int num_fun, HYPRE_Real *mtrx, HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateLaplacian9pt( MPI_Comm comm,
                                         HYPRE_BigInt nx, HYPRE_BigInt ny,
                                         HYPRE_Int P, HYPRE_Int Q,
                                         HYPRE_Int p, HYPRE_Int q,
                                         HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateLaplacian27pt( MPI_Comm comm,
                                          HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                                          HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                          HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                          HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateRotate7pt( MPI_Comm comm,
                                      HYPRE_BigInt nx, HYPRE_BigInt ny,
                                      HYPRE_Int P, HYPRE_Int Q,
                                      HYPRE_Int p, HYPRE_Int q,
                                      HYPRE_Real alpha, HYPRE_Real eps );
HYPRE_ParCSRMatrix GenerateDifConv( MPI_Comm comm,
                                    HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                                    HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                    HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                    HYPRE_Real *value );
HYPRE_ParCSRMatrix GenerateVarDifConv( MPI_Comm comm,
                                       HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                                       HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                       HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                       HYPRE_Real eps, HYPRE_ParVector *rhs_ptr );
HYPRE_ParCSRMatrix GenerateRSVarDifConv( MPI_Comm comm,
                                         HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                                         HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                                         HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                                         HYPRE_Real eps, HYPRE_ParVector *rhs_ptr,
                                         HYPRE_Int type );
float * GenerateCoordinates( HYPRE_BigInt nx, HYPRE_BigInt ny, HYPRE_BigInt nz,
                             HYPRE_Int P, HYPRE_Int Q, HYPRE_Int R,
                             HYPRE_Int p, HYPRE_Int q, HYPRE_Int r,
                             HYPRE_Int coorddim );
