/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

      integer HYPRE_ERROR_GENERIC
      integer HYPRE_ERROR_MEMORY
      integer HYPRE_ERROR_ARG
      integer HYPRE_ERROR_CONV
      parameter (HYPRE_ERROR_GENERIC = 1)
      parameter (HYPRE_ERROR_MEMORY  = 2)
      parameter (HYPRE_ERROR_ARG     = 4)
      parameter (HYPRE_ERROR_CONV    = 256)
