#!/usr/bin/env octave -q

id = str2num(argv(){1});
fprintf('Process ID used: %i\n', id)

calculate_constant;
