#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

#=============================================================================
# EXAMPLES: Compare ex*.base files with complex.out.* files from current runs
#           differences (except for timings) indicate errors
#=============================================================================

diff -U3 -bI"time" ex18comp.base complex.out.1 >&2
