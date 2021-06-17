#! /bin/bash

# Executable to complete sub files. Must be edited for each use


file=completeNLL.sub


K_VAL=(1 2 5 10 20 100)
BAS_Size=4
ITER=20000
REPEAT=25


for k in ${K_VAL[@]}
do
	echo -e "Arguments\t\t= \"\$(Step) \$(Path) \$(Step) $BAS_Size $k $ITER\"" >> $file
	echo -e "Log\t\t\t= log/complete_CD-${k}_\$(Step).log" >> $file
	echo -e "Error\t\t\t= error/complete_CD-${k}_\$(Step).err" >> $file
	echo -e "Output\t\t\t= out/complete_CD-${k}_\$(Step).out" >> $file
	echo -e "transfer_output_files\t= \$(Path)nll_progress_complete_k${k}-run\$(Step).csv" >> $file
	echo -e "Queue $REPEAT" >> $file
	echo -e "" >> $file
done

