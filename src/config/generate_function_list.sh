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

# Set arguments to "sed" based on the operating system for portability purposes
OS_TYPE="$(uname -s)"
case ${OS_TYPE} in
    Darwin)
        SED_ARGS=(-e 's/^_//' -e 's/_$//')
        ;;

    Linux)
        SED_ARGS=(-e 's/_$//')
        ;;

    *)
        echo "Unknown OS: ${OS_TYPE}"
        exit 1
        ;;
esac

# Use awk to avoid issues with spacing
nm -P *.o *.obj | awk '$2 == "T" {print $1}' | sed "${SED_ARGS[@]}"
