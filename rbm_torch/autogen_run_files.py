import os
import subprocess as sp
import argparse
# Generates  sbatch scripts for all runfiles

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Slurm Files for pytorch RBM and CRBM")
    parser.add_argument('-p', '--partition', nargs="?", type=str, help="Which Partition should the job be submitted on?", default="wzhengpu1")
    parser.add_argument('-q', '--queue', type=str, nargs="?", help="Which Queue should the job be submitted on?", default="wildfire")
    args = parser.parse_args()

    for root, dirs, files in os.walk("../datasets/"):
        if "run_files" in root:
            for file in files:
                sp.check_call(f"python submit_run.py -r {os.path.join(root, file)} -p {args.partition} -q {args.queue}", shell=True)
