#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

hypre_version="2.27.0"
hypre_reldate="2022/12/20"

hypre_major=`echo $hypre_version | cut -d. -f 1`
hypre_minor=`echo $hypre_version | cut -d. -f 2`
hypre_patch=`echo $hypre_version | cut -d. -f 3`

let hypre_number="$hypre_major*10000 + $hypre_minor*100 + $hypre_patch"

