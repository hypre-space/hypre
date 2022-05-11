#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


ex=ex5
dir=`basename \`pwd\``
keys=Aaamc

if [ "$dir" = "vis" ]; then
   dir=.
   mesh=$ex.mesh
   sol=$ex.sol
else
   dir=vis
   mesh=vis/$ex.mesh
   sol=vis/$ex.sol
fi

if [ ! -e $mesh ]
then
   echo "Can't find visualization data for $ex!"
   exit
fi

echo "FiniteElementSpace" > $sol
echo "FiniteElementCollection: H1_2D_P1" >> $sol
echo "VDim: 1" >> $sol
echo "Ordering: 0" >> $sol
echo "" >> $sol
find $dir -name "$ex.sol.??????" | sort | xargs cat >> $sol

glvis -m $mesh -g $sol -k $keys
