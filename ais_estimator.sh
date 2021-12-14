#!/bin/bash

# NOTE: If you bring the matlab files to the same folder as the rest of the code, you can activate this
singleFolder=0;

# If you use an alias for your matlab/octave commands, this should be needed
source "$HOME"/.bashrc
shopt -s expand_aliases

pID=$1;

rm "lnZ_$pID.txt" 2>/dev/null;

if [ $singleFolder = 0 ]; then
  # echo "Changing folder";
  mv "tmp_$pID.rbm" code_AIS/;
  cd code_AIS || exit;
fi

echo "Calling matlab! Using ID $pID"

matlab -nodisplay -nosplash -r "pid=$pID; calculate_constant; exit";
# octave --no-gui octave_init_calcConst.m $pID;

rm "tmp_$pID.rbm";
if [ $singleFolder = 0 ]; then
  mv "lnZ_$pID.txt" ..;
  cd ..;
fi
