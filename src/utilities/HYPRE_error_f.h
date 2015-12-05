cBHEADER**********************************************************************
c Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
c Produced at the Lawrence Livermore National Laboratory.
c This file is part of HYPRE.  See file COPYRIGHT for details.
c
c HYPRE is free software; you can redistribute it and/or modify it under the
c terms of the GNU Lesser General Public License (as published by the Free
c Software Foundation) version 2.1 dated February 1999.
c
c $Revision: 2.3 $
cEHEADER**********************************************************************

      integer HYPRE_ERROR_GENERIC
      integer HYPRE_ERROR_MEMORY
      integer HYPRE_ERROR_ARG
      integer HYPRE_ERROR_CONV
      parameter (HYPRE_ERROR_GENERIC = 1)
      parameter (HYPRE_ERROR_MEMORY  = 2)
      parameter (HYPRE_ERROR_ARG     = 4)
      parameter (HYPRE_ERROR_CONV    = 256)
