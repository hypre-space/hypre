#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************

#=============================================================================
# EXAMPLES: Compare ex*.base files with complex.out.* files from current runs
#           differences (except for timings) indicate errors
#=============================================================================

diff -U3 -bI"time" ex18comp.base complex.out.1 >&2
