#!/bin/python
import argparse
from analysis.global_info import get_global_info
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Slurm Files for pytorch RBM and CRBM")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-d', '--datatype', help='Datatype Specifier String. See global_info', required=True)
    requiredNamed.add_argument('-r', '--round', type=str, help="Which Round should we run the scripts on?", required=True)
    requiredNamed.add_argument('-p', '--partition', type=str, help="Which Partition should the job be submitted on?", required=True)
    requiredNamed.add_argument('-q', '--queue', type=str, help="Which Queue should the job be submitted on?", required=True)
    requiredNamed.add_argument('-m', '--model', type=str, help="Which Model are we training, i.e. rbm or crbm?", required=True)
    requiredNamed.add_argument('-e', '--epochs', type=str, help="Number of Training Iterations", required=True)
    requiredNamed.add_argument('-g', '--gpus', type=str, help="Number of gpus available", required=True)
    parser.add_argument('--wdir', nargs="?", type=str, help="Manually Set working directory, Usually handled internally.")
    parser.add_argument('--precision', type=str, help="Set precision of the model, single or double", default="double")
    parser.add_argument('-c', nargs="?", help="Number of CPU cores to use. Default is 6.", default="6", type=str)
    parser.add_argument('-w', nargs="?", help="Weight File name to use to weight model training. Must be in same directory as the sequence files. Alternatively can be 'fasta' or None", default=None, type=str)
    parser.add_argument('--walltime', default="7-00:00", type=str, nargs="?", help="Set wall time for training")
    parser.add_argument('-a', '--account', default="jprocyk", type=str, nargs="?", help="Account name for server submission")
    parser.add_argument('--email', default="jprocyk@asu.edu", type=str, nargs="?", help="Email for Job notifications")
    parser.add_argument('--error', type=str, help="Set slurm error file prefix", default="slurm", nargs="?")
    args = parser.parse_args()

    # Process Our Arguments
    if args.wdir is None:
        wdir = "/scratch/jprocyk/machine_learning/phage_display_ML/rbm_torch/"
    else:
        wdir = args.wdir

    clusternum = args.round[-1]  # cluster the data belongs to (). Need to use a string to access
    if clusternum.isalpha() or clusternum.isdigit() and args.round[-2] != "c":  # No cluster number, set to 1
        clusternum = str(1)
    elif clusternum.isdigit() and args.round[-2] == "c":  # Cluster specified, get clusternum
        pass
    else:  # Character is neither a letter nor number
        print(f"Cluster Designation {clusternum} is not supported.")
        exit(-1)


    # Now we get the global information which gives us specifics
    try:
        info = get_global_info(args.datatype, dir="../datasets/dataset_files/")
    except KeyError:
        print(f"Key {args.datatype} not found in get_global_info function in /analysis/analysis_methods.py")
        exit(-1)

    # Set up output directory
    datatype_folder = f"./submission/{args.datatype}/"
    if not os.path.exists(datatype_folder):
        os.mkdir(datatype_folder)

    # Make a list of paths for all files we want
    paths = []
    outs = []

    # if "all" in args.round:
    #     paths = [info["data_dir"][3:] + x for x in info["data_files"][clusternum]]
    #     model_names = info["model_names"][clusternum]
    #     if args.w is not None:
    #         model_names = [x + "_" + extension for x in model_names]
    #
    #     outs = [f"{args.datatype}_{args.model}_{x}" for x in info["model_names"][clusternum]]
    #
    # else:

    data_index = info['data_files'][clusternum].index(args.round+".fasta")
    if data_index == -1:
        print(f"Dataset {args.round+'.fasta'} Not Found. Please Ensure everything is listed correctly in global_info")
    paths.append(info["data_dir"][3:] + info["data_files"][clusternum][data_index])

    if args.w == "fasta":
        extension = "f"
    elif args.w is not None:
        try:
            with open(os.path.dirname(paths[-1]) + "/" + args.w) as f:
                data = json.load(f)
        except IOError:
            print(f"Could not load provided weight file {os.path.dirname(paths[-1]) + '/' + args.w}")
            exit(-1)
        extension = data["extension"]

    if args.w is not None:
        outs.append(f"{args.datatype}_{args.model}_{args.round}_{extension}")
    else:
        outs.append(f"{args.datatype}_{args.model}_{args.round}")

    for pid, p in enumerate(paths):
        o = open(f'./submission_templates/sbatch_template.sh', 'r')
        filedata = o.read()
        o.close()

        # Replace the Strings we want
        filedata = filedata.replace("ACCOUNT", args.account)
        filedata = filedata.replace("DATATYPE", args.datatype)
        filedata = filedata.replace("DATAPATH", p)
        filedata = filedata.replace("PARTITION", args.partition)
        filedata = filedata.replace("QUEUE", args.queue)
        filedata = filedata.replace("MODEL", args.model)
        filedata = filedata.replace("OUT", outs[pid])
        filedata = filedata.replace("ERR", args.error)
        filedata = filedata.replace("EPOCHS", args.epochs)
        filedata = filedata.replace("GPUS", args.gpus)
        filedata = filedata.replace("WDIR", wdir)
        filedata = filedata.replace("EMAIL", args.email)
        filedata = filedata.replace("CORES", args.c)
        filedata = filedata.replace("WALLTIME", args.walltime)
        filedata = filedata.replace("PRECISION", args.precision)

        if args.w is not None:
            filedata = filedata.replace("WEIGHTS", args.w)
        else:
            filedata = filedata.replace("WEIGHTS", 'None')

        with open(f"{datatype_folder}{outs[pid]}.sh", 'w+') as file:
            file.write(filedata)

    if len(paths) > 1:
        tmp = args.model
        if info["clusters"] > 1:
            tmp += f"_c{clusternum}"
        if args.w:
            tmp += f"_{extension}"
        with open(f".{datatype_folder}submit_{args.datatype}_{tmp}.sh", 'w+') as file:
            file.write("#!/bin/bash\n")
            for out in outs:
                file.write(f"sbatch {out}.sh \n")
