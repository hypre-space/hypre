! -*- fortran -*-
!BHEADER**********************************************************************
! Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
! Produced at the Lawrence Livermore National Laboratory.
! This file is part of HYPRE.  See file COPYRIGHT for details.
!
! HYPRE is free software; you can redistribute it and/or modify it under the
! terms of the GNU Lesser General Public License (as published by the Free
! Software Foundation) version 2.1 dated February 1999.
!
! $Revision: 2.3 $
!EHEADER**********************************************************************


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

