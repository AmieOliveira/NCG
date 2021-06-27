#! /bin/bash

# Executable to complete sub files. Must be edited for each use


filename=completeNLL
baseId=complete

K_VAL=(5 10 20 100)
BAS_Size=4
ITER=20000
REPEAT=25

basePath=$PWD
file=result/${baseId}/${filename}.sub


echo -e "###############################\n# ${baseId} net NLL eval\n###############################\n" >> $file
echo -e "Executable\t\t= ${basePath}/complete.exe" >> $file
echo -e "Universe\t\t= vanilla" >> $file
echo -e "should_transfer_files\t= IF_NEEDED" >> $file
echo -e "when_to_transfer_output\t= ON_EXIT" >> $file
echo -e "\n" >> $file


for k in ${K_VAL[@]}
do
	echo -e "Arguments\t\t= \"\$(Step) . \$(Step) $BAS_Size $k $ITER\"" >> $file
	echo -e "Log\t\t\t= ${basePath}/log/${baseId}_CD-${k}.log" >> $file
	echo -e "Error\t\t\t= ${basePath}/error/${baseId}_CD-${k}_\$(Step).err" >> $file
	echo -e "Output\t\t\t= ${basePath}/out/${baseId}_CD-${k}_\$(Step).out" >> $file
	echo -e "transfer_output_files\t= nll_progress_${baseId}_k${k}-run\$(Step).csv" >> $file

	echo -e "Queue $REPEAT" >> $file
	echo -e "" >> $file
done

