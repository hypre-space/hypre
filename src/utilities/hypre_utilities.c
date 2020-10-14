/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

HYPRE_Int hypre_multmod(HYPRE_Int a, HYPRE_Int b, HYPRE_Int mod)
{
    HYPRE_Int res = 0; // Initialize result
    a %= mod;
    while (b)
    {
        // If b is odd, add a with result
        if (b & 1)
        {
            res = (res + a) % mod;
        }
        // Here we assume that doing 2*a
        // doesn't cause overflow
        a = (2 * a) % mod;
        b >>= 1;  // b = b / 2
    }
    return res;
}

