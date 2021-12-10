#!/bin/bash

# NOTE: If you bring the matlab files to the same folder as the rest of the code, you can activate this
singleFolder=0;

# If you use an alias for your matlab/octave commands, this should be needed
source "$HOME"/.bashrc
shopt -s expand_aliases

rm lnZ.txt 2>/dev/null;

if [ $singleFolder = 0 ]; then
  # echo "Changing folder";
  mv tmp.rbm code_AIS/;
  cd code_AIS || exit;
fi

matlab -nodisplay -nosplash < calculate_constant.m;
# octave --no-gui calculate_constant.m;

if [ $singleFolder = 0 ]; then
  rm tmp.rbm;
  mv lnZ.txt ..;
  cd ..;
fi
