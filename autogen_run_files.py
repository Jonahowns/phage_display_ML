import os
import subprocess as sp
import argparse
# Generates  sbatch scripts for all runfiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Slurm Files for pytorch RBM and CRBM")
    parser.add_argument('-p', '--partition', nargs="?", type=str, help="Which Partition should the job be submitted on?", default="wzhengpu1")
    parser.add_argument('-q', '--queue', type=str, nargs="?", help="Which Queue should the job be submitted on?", default="wildfire")
    parser.add_argument('-d', nargs="?", type=str, help="delete all existing slurm sbatch files", default="False")
    args = parser.parse_args()

    # delete all existing submission scripts in submission sub directories
    if args.d == "True":
        for root, dirs, files in os.walk("./datasets/"):
            if "submission" in root:
                for file in files:
                    if file[-3:] == ".sh":
                        sp.check_call(f"rm {os.path.join(root, file)}", shell=True)

    # Generate submission files from run files
    for root, dirs, files in os.walk("./datasets/"):
        if "run_files" in root:
            for file in files:
                sp.check_call(f"python submit_run.py -r {os.path.join(root, file)} -p {args.partition} -q {args.queue}", shell=True)


