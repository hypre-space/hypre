!     Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
!     HYPRE Project Developers. See the top-level COPYRIGHT file for details.
!
!     SPDX-License-Identifier: (Apache-2.0 OR MIT)

! -*- fortran -*-
!******************************************************************************
! 
!  Header file for HYPRE library
! 
! ****************************************************************************



! --------------------------------------------------------------------------
!  Structures
! --------------------------------------------------------------------------

! --------------------------------------------------------------------------
!  Constants
! --------------------------------------------------------------------------

      integer HYPRE_UNITIALIZED
      parameter( HYPRE_UNITIALIZED = -999 )

      integer HYPRE_PETSC_MAT_PARILUT_SOLVER
      parameter( HYPRE_PETSC_MAT_PARILUT_SOLVER = 222 )
      integer HYPRE_PARILUT
      parameter( HYPRE_PARILUT =                  333 )

      integer HYPRE_STRUCT
      parameter( HYPRE_STRUCT =  1111 )
      integer HYPRE_SSTRUCT
      parameter( HYPRE_SSTRUCT = 3333 )
      integer HYPRE_PARCSR
      parameter( HYPRE_PARCSR =  5555 )

      integer HYPRE_ISIS
      parameter( HYPRE_ISIS =    9911 )
      integer HYPRE_PETSC
      parameter( HYPRE_PETSC =   9933 )

      integer HYPRE_PFMG
      parameter( HYPRE_PFMG =    10 )
      integer HYPRE_SMG
      parameter( HYPRE_SMG =     11 )

