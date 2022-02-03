/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "HYPRE_config.h"

#if defined(HYPRE_USING_SYCL)

/* oneAPI DPL headers */
/* NOTE: these must be included before standard C++ headers */

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/functional>

#endif
