import os

path = "/home/ss_students/mtp/modality_pipeline_wft_v3-main/utils_custom/monai_losses.py"
with open(path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("device = torch.device"):
        new_lines.append("if __name__ == '__main__':\n")
        new_lines.append("    " + line)
    elif line.startswith("perc_loss = ") or line.startswith("l1 = ") or line.startswith("lambda_perc = "):
        new_lines.append("    " + line)
    elif line.startswith("def loss_fn"):
        new_lines.append("    " + line)
    elif line.strip().startswith("return l1"):
        new_lines.append("        " + line.strip() + "\n")
    else:
        new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)

print("monai_losses.py successfully updated!")
