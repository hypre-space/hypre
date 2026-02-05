#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Print the defined function names in the object files of the current directory.
#
# The script uses 'nm' and searches for functions labeled with 'T', i.e.,
# symbol is in the text (code) section and is globally visible.

# This prevents unmatched patterns from expanding (e.g., when there are no .obj files)
shopt -s nullglob

# Use awk to avoid issues with spacing
# Demangle any c++ name mangling and filter _device_stub_ prefixes.
nm -P *.o *.obj | awk '$2 == "T" {print $1}' | c++filt | sed -e 's/(.*$//' -e 's/^__device_stub__//' -e 's/^_//' -e 's/_$//'

