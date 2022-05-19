supported_ml_models = ["rbm", "crbm"]


# Datatype defines the basics of our data, Each datatype is specified for a group of related fasta files
# Focus - > short string specifier that gives the overall dataset we are using
# Molecule -> What kind of sequence data? currently protein, dna, and rna are supported
# id -> short string specifier ONLY for datasets which have different clustering methods
# process -> How were the gaps added to each dataset
# clusters -> How many clusters are in each data file
# cluster_indices -> Define the lengths of data put in each cluster, It is inclusive so [12, 16] includes length 12 and length 16. There must be cluster_indices for each cluster
# gap_position_indices -> Index where gaps should be added to each sequence that is short of the maximum length.
pig_ge2_datatype = {"focus": "pig", "molecule": "protein", "id": "ge2", "process": "gaps_end", "clusters": 2, "gap_position_indices": [-1, -1], "cluster_indices": [[12, 22], [35, 45]]}
pig_gm2_datatype = {"focus": "pig", "molecule": "protein", "id": "gm2", "process": "gaps_middle", "clusters": 2, "gap_position_indices": [2, 16], "cluster_indices": [[12, 22], [35, 45]]}  # based solely off looking at the sequence logos
pig_ge4_datatype = {"focus": "pig", "molecule": "protein", "id": "ge4", "process": "gaps_end", "clusters": 4, "gap_position_indices": [-1, -1, -1, -1], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}
pig_gm4_datatype = {"focus": "pig", "molecule": "protein", "id": "gm4", "process": "gaps_middle", "clusters": 4, "gap_position_indices": [2, 2, 16, 16], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}
cov_datatype = {"focus": "cov", "molecule": "dna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[40, 40]]}

supported_datatypes = {"pig_ge2": pig_ge2_datatype,
                    "pig_gm2": pig_gm2_datatype,
                    "pig_ge4": pig_ge4_datatype,
                    "pig_gm4": pig_gm4_datatype,
                    "cov": cov_datatype}


def get_global_info(datatype_str, cluster=0, weights=False, model="rbm"):
    if model not in supported_ml_models:
        print(f"Model {model} not supported. Please edit /rbm_torch/analysis/global_info.py to properly enable support for other ml models.")
        exit(-1)

    try:
        datatype = supported_datatypes[datatype_str]
    except KeyError:
        print(f"Datatype {datatype_str} not found. Please add required data to get_global_info function in /rbm_torch/analysis/global_info.py")
        exit(-1)


    if type(cluster) == int and cluster > datatype["clusters"]:
        print(f"Cluster {cluster} does not exist for datatype {datatype_str}")
        exit(-1)
    elif type(cluster) == str and cluster != "all":
        print(f"Supported cluster options include the cluster number or 'all'. Received {cluster}")
        exit(-1)

    # Which directory are trained models stored in locally (This is for my globus transfer script to function properly)
    local_model_location = {"pig_gm2": f"/mnt/D1/globus/pig_trained_{model}s/gm2/",
                   "pig_ge2": f"/mnt/D1/globus/pig_trained_{model}s/ge2/",
                   "pig_gm4": f"/mnt/D1/globus/pig_trained_{model}s/gm4/",
                   "pig_ge4": f"/mnt/D1/globus/pig_trained_{model}s/ge4/",
                   "cov": f"/mnt/D1/globus/cov_trained_{model}s/"}[datatype_str]

    # Paths from current directory (analysis) to location of data files and trained models folder
    data_location = {"pig_ge2": "../../pig_tissue/gaps_end_2_clusters/",
                "pig_gm2": "../../pig_tissue/gaps_middle_2_clusters/",
                "pig_gm4": "../../pig_tissue/gaps_middle_4_clusters/",
                "pig_ge4": "../../pig_tissue/gaps_end_4_clusters/",
                "invivo": "../../invivo/",
                "rod": "../../rod/",
                "cov": "../../cov/"}[datatype_str]

    # Server model location provides relative path from phage_display_ML directory to the server-side trained model directory
    server_model_location = data_location[6:] + f"trained_{model}s/"

    if datatype["focus"] == "pig":
        molecule = "protein"
        rounds = ['np1', 'np2', 'np3', 'n1', 'b3']
        if type(cluster) == str and cluster == "all":
            all_data_files, all_model_names, all_rounds, all_configkeys = [], [], [], []
            for i in range(datatype["clusters"]):
                all_data_files.append([x + f'_c{i+1}.fasta' for x in rounds])
                all_configkeys.append(f"{datatype_str[:4]}c{i+1}_{datatype_str[4:]}")
                if weights:
                    all_model_names.append([x + f"_c{i+1}_w" for x in rounds])
                    all_rounds.append([x + f"_c{i+1}" for x in rounds])
                else:
                    all_model_names.append([x + f"_c{i+1}" for x in rounds])
                    all_rounds.append([x + f"_c{i+1}" for x in rounds])

            info = {"data_files": all_data_files, "rounds": all_rounds,
                    "model_names": all_model_names, "local_model_dir": local_model_location,
                    "data_dir": data_location, "server_model_dir": server_model_location,
                    "molecule": molecule, "cluster": "all", "configkeys": all_configkeys}
        else:
            assert cluster > 0
            all_data_files = [x + f'_c{cluster}.fasta' for x in rounds]
            if weights:
                all_model_names = [x + f"_c{cluster}_w" for x in rounds]
                all_rounds = [x + f"_c{cluster}" for x in rounds]
            else:
                all_model_names = [x + f"_c{cluster}" for x in rounds]
                all_rounds = all_model_names

            info = {"data_files": all_data_files, "rounds": all_rounds,
                    "model_names": all_model_names, "local_model_dir": local_model_location,
                    "data_dir": data_location, "server_model_dir": server_model_location,
                    "molecule": molecule, "cluster": cluster, "configkey": f"{datatype_str[:4]}c{cluster}_{datatype_str[4:]}"}

    elif datatype["focus"] == "cov":
        molecule = "dna"

        all_data_files = [f"r{i}.fasta" for i in range(1, 13)]
        if weights:
            all_rounds = [f"r{i}_w" for i in range(1, 13)]
        else:
            all_rounds = [f"r{i}" for i in range(1, 13)]

        all_model_names = all_rounds

        info = {"data_files": all_data_files, "rounds": all_rounds,
                "model_names": all_model_names, "local_model_dir": local_model_location,
                "data_dir": data_location, "server_model_dir": server_model_location,
                "molecule": molecule, "configkey": f"{datatype_str[:3]}"}

    elif datatype["focus"] == "invivo":
        molecule = "protein"
        # dest_path = f"../invivo/trained_{model}s/"
        # src_path = "../invivo/"
        #
        # c1 = [x + '_c1.fasta' for x in rounds]
        # c2 = [x + '_c2.fasta' for x in rounds]
        #
        # all_data_files = c1 + c2
        #
        # model1 = [x + '_c1' for x in rounds]
        # model2 = [x + '_c2' for x in rounds]
        #
        # all_model_names = model1 + model2
        #
        # script_names = ["invivo" + str(i) for i in range(len(all_model_names))]
        #
        # paths_to_data = [src_path + x for x in all_data_files]

    return info