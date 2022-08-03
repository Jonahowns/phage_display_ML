import os
import subprocess as sp

# Generates  sbatch scripts for all runfiles

if __name__ == '__main__':
    for root, dirs, files in os.walk("../datasets/"):
        if "run_files" in root:
            for file in files:
                sp.check_call(f"python submit_run.py -r {os.path.join(root, file)}", shell=True)
