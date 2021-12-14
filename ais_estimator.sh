#!/bin/bash

# NOTE: If you bring the matlab files to the same folder as the rest of the code, you can activate this
singleFolder=0;

# If you use an alias for your matlab/octave commands, this should be needed
source "$HOME"/.bashrc
shopt -s expand_aliases

rm "lnZ_$PPID.txt" 2>/dev/null;

if [ $singleFolder = 0 ]; then
  # echo "Changing folder";
  mv "tmp_$PPID.rbm" code_AIS/;
  cd code_AIS || exit;
fi

echo "Calling matlab! Using ID $PPID"

matlab -nodisplay -nosplash -r "pid=$PPID; calculate_constant; exit";
# octave --no-gui calculate_constant.m $PPID;

if [ $singleFolder = 0 ]; then
  rm "tmp_$PPID.rbm";
  mv "lnZ_$PPID.txt" ..;
  ls;
fi
