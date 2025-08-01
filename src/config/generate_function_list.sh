#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Print the defined function names in the object files of the current directory.
#
# The script uses 'nm' and searches for functions labeled with 'T'.

# This prevents unmatched patterns from expanding (e.g., when there are no .obj files)
shopt -s nullglob

nm -P *.o *.obj | grep ' T ' | awk '{print $1}' | sed 's/^_//' | sed 's/_$//'
