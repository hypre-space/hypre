cBHEADER**********************************************************************
c Copyright (c) 2007,  Lawrence Livermore National Laboratory, LLC.
c Produced at the Lawrence Livermore National Laboratory.
c Written by the HYPRE team. UCRL-CODE-222953.
c All rights reserved.
c
c This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
c Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
c disclaimer, contact information and the GNU Lesser General Public License.
c
c HYPRE is free software; you can redistribute it and/or modify it under the
c terms of the GNU General Public License (as published by the Free Software 
c Foundation) version 2.1 dated February 1999.
c
c HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
c WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
c FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
c Public License for more details.
c
c You should have received a copy of the GNU Lesser General Public License
c along with this program; if not, write to the Free Software Foundation,
c Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
c
c $Revision$
cEHEADER**********************************************************************

      integer HYPRE_ERROR_GENERIC
      integer HYPRE_ERROR_MEMORY
      integer HYPRE_ERROR_ARG
      integer HYPRE_ERROR_CONV
      parameter (HYPRE_ERROR_GENERIC = 1)
      parameter (HYPRE_ERROR_MEMORY  = 2)
      parameter (HYPRE_ERROR_ARG     = 4)
      parameter (HYPRE_ERROR_CONV    = 256)
