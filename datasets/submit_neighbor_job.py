#!/bin/python
import argparse
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Slurm Files for pytorch RBM and CRBM")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-d', '--datadir', help='directory of fasta file', required=True)
    requiredNamed.add_argument('-f', '--fastafile', type=str, nargs="+", help="fasta file", required=True, default=[])
    requiredNamed.add_argument('-m', '--molecule', type=str, help="Which molecule?", required=True)
    requiredNamed.add_argument('-o', '--out', type=str, help="File Name for generated sbatch file", required=True)
    requiredNamed.add_argument('-t1', type=str, help="1st threshold value", required=True)
    requiredNamed.add_argument('-t2', type=str, help="2nd threshold value", required=True)
    parser.add_argument('-c', nargs="?", help="Number of CPU cores to use. Default is 6.", default="6", type=str)

    args = parser.parse_args()

    # Get header info and modify
    o = open(f'./submit_neighbors/neighbor_header.sh', 'r')
    header = o.read()
    o.close()

    header = header.replace("THREADS", args.c)

    out_file = open(f"./submit_neighbors/{args.out}.sh", 'w+')
    out_file.write(header)

    for fid, f in enumerate(args.fastafile):

        launch_string = "python pairwise_distances.py DATA_DIR FASTA_FILE MOLECULE THREADS THRESH1 THRESH2\n"
        tmp = copy.deepcopy(launch_string)

        # Replace the Strings we want
        tmp = tmp.replace("DATA_DIR", args.datadir)
        tmp = tmp.replace("FASTA_FILE", f)
        tmp = tmp.replace("MOLECULE", args.molecule)
        tmp = tmp.replace("THREADS", args.c)
        tmp = tmp.replace("THRESH1", args.t1)
        tmp = tmp.replace("THRESH2", args.t2)

        out_file.write(tmp)

    out_file.close()
