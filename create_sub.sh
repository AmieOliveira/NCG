#! /bin/bash

# Executable to complete sub files. Must be edited for each use


filename=conv-mnist
baseId=mnist

K_VAL=(1 5 10)
#(1 2 5 10 20 100)
# BAS_Size=4
ITER=20
REPEAT=5
# V_VAL=(14 12 10 8 6 4)
# V_TYPE=l
# V_TYPE=s
# VERSIONS=(2 3 4)
LR_VAL=(0.01)
# P_VAL=(1 0.75 0.5 0.25)

trainType=convolution
trainParam=0

H=500
BATCH=50
F_NLL=1

basePath=$PWD
file=result/${baseId}/${filename}.sub
# file=result/lRate05/${filename}.sub

# if [ $V_TYPE = "l" ]; then
# 	V_TYPE_LONG=line
# else 
# 	if [ $V_TYPE = "s" ]; then
# 		V_TYPE_LONG=spiral
# 	fi
# fi

echo -e "###############################\n# ${baseId} opt NLL eval\n###############################\n" >> $file
echo -e "Executable\t\t= ${basePath}/${baseId}.exe" >> $file
echo -e "Universe\t\t= vanilla" >> $file
echo -e "should_transfer_files\t= IF_NEEDED" >> $file
echo -e "when_to_transfer_output\t= ON_EXIT" >> $file

echo -e "transfer_input_files\t= /home/users/amandacno/ConnectivityPatterns/Datasets/bin_mnist-train.data" >> $file

echo -e "\n" >> $file


for k in ${K_VAL[@]}
do
	for lr in ${LR_VAL[@]}  
		# ${V_VAL[@]} ${VERSIONS[@]}
	do
		# for p in ${P_VAL[@]}
		# do
		# 	echo -e "Arguments\t\t= \"\$(Step) . \$(Step) $BAS_Size $k $ITER 5 $lr $p 1\"" >> $file
		# 	echo -e "Log\t\t\t= ${basePath}/log/bas${BAS_Size}_${baseId}.log" >> $file
		# 	echo -e "Error\t\t\t= ${basePath}/error/bas${BAS_Size}_${baseId}_CD-${k}_lr${lr}_p${p}_\$(Step).err" >> $file
		# 	echo -e "Output\t\t\t= ${basePath}/out/bas${BAS_Size}_${baseId}${v}_CD-${k}_lr${lr}_p${p}_\$(Step).out" >> $file
		# 	echo -e "transfer_output_files\t= nll_bas${BAS_Size}_${baseId}_CD-${k}_lr${lr}_p${p}_run\$(Step).csv,connectivity_bas${BAS_Size}_${baseId}_CD-${k}_lr${lr}_p${p}_run\$(Step).csv" >> $file
		# 
		# 	echo -e "Queue $REPEAT" >> $file
		# 	echo -e "" >> $file

		echo -e "Arguments\t\t= \"\$(Step) . \$(Step) $trainType $trainParam $k $ITER $H $BATCH $lr $F_NLL\"" >> $file
		echo -e "Log\t\t\t= ${basePath}/log/${baseId}.log" >> $file
		echo -e "Error\t\t\t= ${basePath}/error/${baseId}_${trainType}_CD-${k}_lr${lr}_\$(Step).err" >> $file
		echo -e "Output\t\t\t= ${basePath}/out/${baseId}_${trainType}_CD-${k}_lr${lr}_\$(Step).out" >> $file
		echo -e "transfer_output_files\t= ${baseId}_${trainType}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}_run\$(Step).rbm,nll_${baseId}_${trainType}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}_run\$(Step).csv" >> $file
		# ,connectivity_${baseId}_${trainType}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}_run\$(Step).csv
		
		echo -e "Queue $REPEAT" >> $file
		echo -e "" >> $file
		# done
	done
done

echo "File created"

