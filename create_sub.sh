#! /bin/bash

# Executable to complete sub files. Must be edited for each use


filename=spiralNeigh
baseId=neighbors

K_VAL=(1 2 5 10 20 100)
BAS_Size=4
ITER=10000
REPEAT=25
V_VAL=(14 12 10 8 6 4)
# V_TYPE=l
V_TYPE=s
# VERSIONS=(2 3)

basePath=$PWD
file=result/${baseId}/${filename}.sub
# file=result/lRate05/${filename}.sub

if [ $V_TYPE = "l" ]; then
	V_TYPE_LONG=line
else 
	if [ $V_TYPE = "s" ]; then
		V_TYPE_LONG=spiral
	fi
fi

echo -e "###############################\n# ${baseId} net NLL eval\n###############################\n" >> $file
echo -e "Executable\t\t= ${basePath}/${baseId}.exe" >> $file
echo -e "Universe\t\t= vanilla" >> $file
echo -e "should_transfer_files\t= IF_NEEDED" >> $file
echo -e "when_to_transfer_output\t= ON_EXIT" >> $file
echo -e "\n" >> $file


for k in ${K_VAL[@]}
do
	for v in ${V_VAL[@]}  
		# ${VERSIONS[@]}
	do
		echo -e "Arguments\t\t= \"\$(Step) . \$(Step) $BAS_Size $v $V_TYPE $k $ITER 5 0.01 1\"" >> $file
		echo -e "Log\t\t\t= ${basePath}/log/bas${BAS_Size}_${baseId}.log" >> $file
		echo -e "Error\t\t\t= ${basePath}/error/bas${BAS_Size}_${baseId}${v}-${V_TYPE}_CD-${k}_\$(Step).err" >> $file
		echo -e "Output\t\t\t= ${basePath}/out/bas${BAS_Size}_${baseId}${v}-${V_TYPE}_CD-${k}_\$(Step).out" >> $file
		echo -e "transfer_output_files\t= nll_progress_bas${BAS_Size}_${baseId}${v}_${V_TYPE_LONG}_k${k}-run\$(Step).csv" >> $file

		echo -e "Queue $REPEAT" >> $file
		echo -e "" >> $file
	done
done

