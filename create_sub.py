"""
    Script to create HTCondor submission files
"""

# TODO: Por enquanto está tudo para o script de acurácia (tempo curto, etc.)

from pathlib import Path


executable = "accuracy"
dataset = "mnist"

extras = ""

trainType = "complete"

k_vals = [1, 10]
H = 500
lRate = 0.1
bSize = 50
iterations = 100
freq = 10
repeat = 5

p_vals = [1, 0.5, 0.1]

basepath = Path().absolute()
output = f"result/{dataset}/{trainType}-{executable}{extras}.sub"

print(f"output = {basepath}/{output}")

with open(f"{basepath}/{output}", "w") as f:
    f.write(f"###############################\n# {dataset.upper()} opt classif eval\n###############################\n\n")
    f.write(f"Executable\t\t= {basepath}/{executable}.exe\n")
    f.write(f"initialdir\t\t= {basepath}/result/{dataset}\n")
    f.write("Universe\t\t= vanilla\n")
    f.write("should_transfer_files\t= YES\n")
    f.write("when_to_transfer_output\t= ON_EXIT\n")

    f.write(f"transfer_input_files\t= {basepath}/Datasets/bin_mnist-train.data,{basepath}/Datasets/bin_mnist-test.data\n")
    f.write(f"transfer_output_remaps\t= \"{basepath}/Datasets/bin_mnist-train.data = Datasets/bin_mnist-train.data\"\n")
    f.write(f"transfer_output_remaps\t= \"{basepath}/Datasets/bin_mnist-test.data = Datasets/bin_mnist-test.data\"\n")
    
    f.write(f"\n\n")

    for k in k_vals:
        if trainType in ["complete", "convolution"]:
            basename = f"{dataset}_{trainType}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{iterations}_withLabels_run"

            f.write(f"Arguments\t\t= \"$(Step) . $(Step) {trainType} 0 {k} {iterations} {H} {bSize} {lRate} {freq}\n")
            f.write(f"Log\t\t\t= {basepath}/log/{executable}.log\n")
            f.write(f"Error\t\t\t= {basepath}/error/{executable}_{trainType}_CD-{k}_lr{lRate}_$(Step).err\n")
            f.write(f"Output\t\t\t= {basepath}/out/{executable}_{trainType}_CD-{k}_lr{lRate}_$(Step).out\n")
            f.write(f"transfer_output_files\t= acc_{basename}$(Step).csv\n")

            f.write(f"Queue {repeat}\n")
            f.write(f"\n")

        elif trainType == "sgd":
            for p in p_vals:
                basename = f"{dataset}_{trainType}-{p}_H{H}_CD-{k}_lr{lRate}_mBatch{bSize}_iter{iterations}_withLabels_run"
                
                f.write(f"Arguments\t\t= \"$(Step) . $(Step) {trainType} {p} {k} {iterations} {H} {bSize} {lRate} {freq}\n")
                f.write(f"Log\t\t\t= {basepath}/log/{executable}.log\n")
                f.write(f"Error\t\t\t= {basepath}/error/{executable}_{trainType}-{p}_CD-{k}_lr{lRate}_$(Step).err\n")
                f.write(f"Output\t\t\t= {basepath}/out/{executable}_{trainType}-{p}_CD-{k}_lr{lRate}_$(Step).out\n")
                f.write(f"transfer_output_files\t= acc_{basename}$(Step).csv\n")

                f.write(f"Queue {repeat}\n")
                f.write(f"\n")

print("----> Done")


