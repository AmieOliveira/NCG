#! /bin/bash

# Executable to complete sub files. Must be edited for each use


filename=GONC-lr0.01_H25
baseId=mnist

K_VAL=(10 1)
#(1 2 5 10 20 100)
# BAS_Size=4
ITER=1000
REPEAT=2
# V_VAL=(14 12 10 8 6 4)
# V_TYPE=l
# V_TYPE=s
# VERSIONS=(2 3 4)
LR_VAL=(0.01)
P_VAL=(1 0.5)
# 0.1)

trainType=sgd
trainParam=0

H=25
BATCH=50
F_NLL=100
LABELS=0

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

if [ $LABELS = 1 ]; then
	LABEL_STR=_withLabels
fi

echo -e "###############################\n# ${baseId} opt NLL eval\n###############################\n" >> $file
echo -e "Executable\t\t= ${basePath}/${baseId}.exe" >> $file
echo -e "initialdir\t\t= ${basePath}/result/${baseId}" >> $file
echo -e "Universe\t\t= vanilla" >> $file
echo -e "should_transfer_files\t= YES" >> $file
echo -e "when_to_transfer_output\t= ON_EXIT" >> $file

echo -e "transfer_input_files\t= ${basePath}/result/${baseId},${basePath}/Datasets/bin_mnist-train.data" >> $file
echo -e "transfer_output_remaps\t= \"${basePath}/Datasets/bin_mnist-train.data = bin_mnist-train.data\"" >> $file

echo -e "\n" >> $file


for k in ${K_VAL[@]}
do
	for lr in ${LR_VAL[@]}  
		# ${V_VAL[@]} ${VERSIONS[@]}
	do
		for trainParam in ${P_VAL[@]}
		do
			echo -e "Arguments\t\t= \"\$(Step) . \$(Step) $trainType $trainParam $k $ITER $H $BATCH $lr $F_NLL $LABELS\"" >> $file
                	echo -e "Log\t\t\t= ${basePath}/log/${baseId}.log" >> $file
                	echo -e "Error\t\t\t= ${basePath}/error/${baseId}_${trainType}-${trainParam}_CD-${k}_lr${lr}${LABEL_STR}_iter${ITER}_\$(Step).err" >> $file
                	echo -e "Output\t\t\t= ${basePath}/out/${baseId}_${trainType}-${trainParam}_CD-${k}_lr${lr}${LABEL_STR}_iter${ITER}_\$(Step).out" >> $file
               		echo -e "transfer_output_files\t= ${baseId}_${trainType}-${trainParam}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}${LABEL_STR}_run\$(Step).rbm,nll_${baseId}_${trainType}-${trainParam}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}${LABEL_STR}_run\$(Step).csv,connectivity_${baseId}_${trainType}-${trainParam}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}${LABEL_STR}_run\$(Step).csv" >> $file


		#echo -e "Arguments\t\t= \"\$(Step) . \$(Step) $trainType $trainParam $k $ITER $H $BATCH $lr $F_NLL $LABELS\"" >> $file
		#echo -e "Log\t\t\t= ${basePath}/log/${baseId}.log" >> $file
		#echo -e "Error\t\t\t= ${basePath}/error/${baseId}_${trainType}_CD-${k}_lr${lr}${LABEL_STR}_iter${ITER}_\$(Step).err" >> $file
		#echo -e "Output\t\t\t= ${basePath}/out/${baseId}_${trainType}_CD-${k}_lr${lr}${LABEL_STR}_iter${ITER}_\$(Step).out" >> $file
		#echo -e "transfer_output_files\t= ${baseId}_${trainType}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}${LABEL_STR}_run\$(Step).rbm,nll_${baseId}_${trainType}_H${H}_CD-${k}_lr${lr}_mBatch${BATCH}_iter${ITER}${LABEL_STR}_run\$(Step).csv" >> $file
		
		
		echo -e "Queue $REPEAT" >> $file
		echo -e "" >> $file
		done
	done
done

echo "File created"

