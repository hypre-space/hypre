/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * OrderStat.h header file.
 *
 *****************************************************************************/

#ifndef _ORDERSTAT_H
#define _ORDERSTAT_H

#include "_hypre_utilities.h"

HYPRE_Real randomized_select(HYPRE_Real *a, HYPRE_Int p, HYPRE_Int r, HYPRE_Int i);
void hypre_shell_sort(const HYPRE_Int n, HYPRE_Int x[]);

#endif /* _ORDERSTAT_H */
