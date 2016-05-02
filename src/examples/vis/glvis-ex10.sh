#!/bin/sh

ex=ex10
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

if [ ! -e $mesh.000000 ]
then
   echo "Can't find visualization data for $ex!"
   exit
fi

np=`cat $dir/$ex.data | head -n 1 | awk '{ print $2 }'`

glvis -np $np -m $mesh -g $sol -k $keys

