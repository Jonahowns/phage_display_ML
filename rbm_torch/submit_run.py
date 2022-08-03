#!/bin/python
import argparse
import json
import os

default_wdir = "/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/"
default_acc = "jprocyk"
default_email = "jprocyk@asu.edu"
default_partition = "wzhengpu1"
default_queue = "wildfire"
default_python_env = "exmachina3"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Slurm Files for pytorch RBM and CRBM")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-r', '--runfile', type=str, help="JSON file which contains all pertinent info?", required=True)
    # optional arguments
    parser.add_argument('-p', '--partition', nargs="?", type=str, help="Which Partition should the job be submitted on?", default=default_partition)
    parser.add_argument('-q', '--queue', type=str, nargs="?", help="Which Queue should the job be submitted on?", default=default_queue)
    parser.add_argument('--wdir', nargs="?", type=str, help="Manually Set working directory, Usually handled internally.")
    parser.add_argument('-c', nargs="?", help="Number of CPU cores to use. Default is 6.", default="6", type=str)
    parser.add_argument('-w', nargs="?", help="Weight File name to use to weight model training. Must be in same directory as the sequence files. Alternatively can be 'fasta' or None", default=None, type=str)
    parser.add_argument('--walltime', default="7-00:00", type=str, nargs="?", help="Set wall time for training")
    parser.add_argument('-a', '--account', default=default_acc, type=str, nargs="?", help="Account name for server submission")
    parser.add_argument('--email', default=default_email, type=str, nargs="?", help="Email for Job notifications")
    parser.add_argument('--error', type=str, help="Set slurm error file prefix", default="slurm", nargs="?")
    args = parser.parse_args()

    # Process Our Arguments
    if args.wdir is None:
        wdir = default_wdir
    else:
        wdir = args.wdir


    with open(args.runfile, "r") as f:
        run_data = json.load(f)

    # Set up output directory
    datatype_folder = os.path.join(run_data['data_dir'], "submission")
    out_folder = os.path.join(datatype_folder, "out")
    err_folder = os.path.join(datatype_folder, "err")
    if not os.path.exists(datatype_folder):
        os.mkdir(datatype_folder)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(err_folder):
        os.mkdir(err_folder)


    extension = ""
    if run_data["weights"] == "fasta":
        extension = "_f"
    elif run_data["weights"] != "None":
        try:
            with open(run_data["data_dir"] + run_data["weights"]) as f:
                data = json.load(f)
        except IOError:
            print(f"Could not load provided weight file {run_data['data_dir'] + run_data['weights']}")
            exit(-1)
        extension = "_" + data["extension"]

    out = run_data["model_name"] + extension

    o = open(f'./submission_templates/sbatch_run_template.sh', 'r')
    filedata = o.read()
    o.close()

    # Replace the Strings we want
    filedata = filedata.replace("RUNFILE", args.runfile)

    filedata = filedata.replace("ACCOUNT", args.account)
    filedata = filedata.replace("PARTITION", args.partition)
    filedata = filedata.replace("QUEUE", args.queue)
    filedata = filedata.replace("MODEL", run_data["model_type"])
    filedata = filedata.replace("OUT", os.path.join(out_folder, out))
    filedata = filedata.replace("ERR", os.path.join(err_folder, args.error))
    filedata = filedata.replace("GPUS", str(run_data["gpus"]))
    filedata = filedata.replace("PYTHONENV", default_python_env)

    filedata = filedata.replace("WDIR", wdir)
    filedata = filedata.replace("EMAIL", args.email)
    filedata = filedata.replace("CORES", args.c)
    filedata = filedata.replace("WALLTIME", args.walltime)

    with open(os.path.join(datatype_folder, f"{run_data['model_type']}_{out}.sh"), 'w+') as file:
        file.write(filedata)
