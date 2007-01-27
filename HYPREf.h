! -*- fortran -*-
! **********************************************************************
!  Copyright (c) 2007   The Regents of the University of California.
!  Produced at the Lawrence Livermore National Laboratory.
!  Written by the HYPRE team. UCRL-CODE-222953.
!  All rights reserved.
! 
!  This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
!  Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
!  disclaimer, contact information and the GNU Lesser General Public License.
! 
!  HYPRE is free software; you can redistribute it and/or modify it under the 
!  terms of the GNU General Public License (as published by the Free Software
!  Foundation) version 2.1 dated February 1999.
! 
!  HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
!  WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
!  FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
!  Public License for more details.
! 
!  You should have received a copy of the GNU Lesser General Public License
!  along with this program; if not, write to the Free Software Foundation,
!  Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
! 
!  $Revision$
! **********************************************************************



/******************************************************************************
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

