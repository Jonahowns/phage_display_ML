#!/bin/python
import argparse
from analysis.global_info import get_global_info

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
    parser.add_argument('-w', action="store_true", help="Use sequence count to weight sequences")
    parser.add_argument('--wdir', nargs="?", type=str, help="Manually Set working directory, Usually handled internally.")
    parser.add_argument('-o', '--output', type=str, help="Set slurm output file prefix")
    parser.add_argument('-c', nargs="?", help="Number of CPU cores to use. Default is 6.", default="6", type=str)
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

    if args.output is None:
        output = f"{args.datatype}_{args.round}"
    else:
        output = args.output

    clusternum = 0  # Used on datasets that has been divided into clusters of sequences with similar lengths
    if "pig" in args.datatype:  # Pig dataset is the only one that does this right now
        clusternum = int(args.round[-1])

    # Now we get the global information which gives us specifics
    try:
        info = get_global_info(args.datatype, cluster=clusternum, weights=args.w, model=args.model)
    except KeyError:
        print(f"Key {args.datatype} not found in get_global_info function in /analysis/analysis_methods.py")
        exit(-1)

    # Make a list of paths for all files we want
    paths = []
    outs = []
    if "all" in args.round:
        paths = [info["data_dir"][3:] + x for x in info["data_files"]]
        outs = [f"{args.datatype}_{args.model}_{x}" for x in info["model_names"]]
    else:
        data_index = info['data_files'].index(args.round+".fasta")
        if data_index == -1:
            print(f"Dataset {args.round+'.fasta'} Not Found. Please Ensure everything is listed correctly in global_info")
        paths.append(info["data_dir"][3:] + info["data_files"][data_index])
        outs.append(f"{args.datatype}_{args.model}_{output}" + output)

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

        if args.w:
            filedata = filedata.replace("WEIGHTS", 'True')
        else:
            filedata = filedata.replace("WEIGHTS", 'False')

        with open(f"./submission/{outs[pid]}.sh", 'w+') as file:
            file.write(filedata)

    if len(paths) > 1:
        if clusternum != 0:
            tmp = args.model + f"_c{clusternum}"
        else:
            tmp = args.model
        if args.w:
            tmp += "_w"
        with open(f"./submission/submit_{args.datatype}_{tmp}.sh", 'w+') as file:
            file.write("#!/bin/bash\n")
            for out in outs:
                file.write(f"sbatch {out}.sh \n")
