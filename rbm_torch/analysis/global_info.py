import json
import os

supported_ml_models = ["rbm", "crbm"]


# Datatype defines the basics of our data, Each datatype is specified for a group of related fasta files
# Focus - > short string specifier that gives the overall dataset we are using
# Molecule -> What kind of sequence data? currently protein, dna, and rna are supported
# id -> short string specifier ONLY for datasets which have different clustering methods
# process -> How were the gaps added to each dataset, used to name directory
# clusters -> How many clusters are in each data file
# cluster_indices -> Define the lengths of data put in each cluster, It is inclusive so [12, 16] includes length 12 and length 16. There must be cluster_indices for each cluster
# gap_position_indices -> Index where gaps should be added to each sequence that is short of the maximum length.

pig_ge2_datatype = {"focus": "pig", "molecule": "protein", "id": "ge2", "process": "gaps_end", "clusters": 2, "gap_position_indices": [-1, -1], "cluster_indices": [[12, 22], [35, 45]]}
pig_gm2_datatype = {"focus": "pig", "molecule": "protein", "id": "gm2", "process": "gaps_middle", "clusters": 2, "gap_position_indices": [2, 16], "cluster_indices": [[12, 22], [35, 45]]}  # based solely off looking at the sequence logos
pig_ge4_datatype = {"focus": "pig", "molecule": "protein", "id": "ge4", "process": "gaps_end", "clusters": 4, "gap_position_indices": [-1, -1, -1, -1], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}
pig_gm4_datatype = {"focus": "pig", "molecule": "protein", "id": "gm4", "process": "gaps_middle", "clusters": 4, "gap_position_indices": [2, 2, 16, 16], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}
cov_datatype = {"focus": "cov", "molecule": "dna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[40, 40]]}
ribo_datatype = {"focus": "ribo", "molecule": "rna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[115, 121]]}
thc_datatype = {"focus": "thc", "molecule": "rna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[41, 43]]}
pal_datatype = {"focus": "pal", "molecule": "rna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[39, 41]]}

# datatype_str : very important string specifier. For clustered datasets set as datatype["focus"] + datatype["id"]. For non-clustered datasets set as datatype["focus"]
datatype_list = [
    pig_ge2_datatype,
    pig_gm2_datatype,
    pig_ge4_datatype,
    pig_gm4_datatype,
    cov_datatype,
    ribo_datatype,
    thc_datatype,
    pal_datatype
]

supported_datatypes = {}
for datatype in datatype_list:
    if datatype["id"] is None:
        datatype_str = datatype['focus']
    else:
        datatype_str = f"{datatype['focus']}_{datatype['id']}"

    supported_datatypes[datatype_str] = datatype



def get_global_info(datatype_str, dir="../../datasets/dataset_files/"):
    try:
        datatype = supported_datatypes[datatype_str]
    except KeyError:
        print(f"Datatype {datatype_str} not found. Please add required data to get_global_info function in /rbm_torch/analysis/global_info.py")
        exit(-1)

    try:
        info = load_dataset(datatype_str, dir=dir)
    except IOError:
        print("datatype file not found, Please generate!")
        exit(-1)

    return info



def load_dataset(datatype_str, dir="../../datasets/dataset_files/"):
    with open(dir+datatype_str+".json", "r") as json_file:
        info = json.load(json_file)
    return info


def generate_dataset_file(data_filenames, datatype, destination="../../datasets/dataset_files/"):
    # rounds are assigned by each filename
    rounds = [x.split(".")[0] for x in data_filenames]

    if datatype["process"] is not None:
        data_location = f"../../datasets/{datatype['focus']}/{datatype['id']}/"
    else:
        data_location = f"../../datasets/{datatype['focus']}/"

    server_model_location, local_model_location = {}, {}
    for model in supported_ml_models:
        # Server model location provides relative path from phage_display_ML directory to the server-side trained model directory
        server_model_location[model] = data_location[6:] + f"trained_{model}s/"
        # Paths from current directory (analysis) to location of data files and trained models folder
        local_model_location[model] = f"/mnt/D1/globus/pig_trained_{model}s/{datatype['id']}"

    # dictionaries with cluster number as the key
    all_data_files, all_model_names, all_model_names_w, all_rounds, all_configkeys = {}, {}, {}, {}, {}

    if datatype["clusters"] > 1:
        for i in range(datatype["clusters"]):
            all_data_files[i+1] = [x + f'_c{i + 1}.fasta' for x in rounds]

            all_configkeys[i+1] = f"{datatype['focus']}_c{i + 1}_{datatype['id']}"

            all_model_names[i+1] = [x + f"_c{i + 1}" for x in rounds]
            all_model_names_w[i+1] = [x + f"_c{i + 1}_w" for x in rounds]

            all_rounds[i+1] = [x + f"_c{i + 1}" for x in rounds]

    else:
        all_data_files[1] = [f"{i}.fasta" for i in rounds]

        all_rounds[1] = rounds

        all_model_names[1] = [f"{i}" for i in rounds]
        all_model_names_w[1] = [f"{i}_w" for i in rounds]

        all_configkeys[1] = f"{datatype['focus']}"

    info = {"data_files": all_data_files, "rounds": all_rounds,
            "model_names": {"weights": all_model_names_w, "equal": all_model_names},
            "local_model_dir": local_model_location,
            "data_dir": data_location, "server_model_dir": server_model_location,
            "molecule": datatype['molecule'], "configkey": all_configkeys,
            "clusters": datatype["clusters"]}

    if datatype["id"] is not None:
        datatype_str = f"{datatype['focus']}_{datatype['id']}"
    else:
        datatype_str = f"{datatype['focus']}"

    with open(destination+datatype_str+".json", "w+") as json_file:
        json.dump(info, json_file)





    #
    #     info = {"data_files": all_data_files, "rounds": all_rounds,
    #             "model_names": all_model_names, "local_model_dir": local_model_location,
    #             "data_dir": data_location, "server_model_dir": server_model_location,
    #             "molecule": molecule, "cluster": "all", "configkeys": all_configkeys}
    #
    #     data_files = data_filenames
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # # Which directory are trained models stored in locally (This is for my globus transfer script to function properly)
    # local_model_location = {"pig_gm2": f"/mnt/D1/globus/pig_trained_{model}s/gm2/",
    #                "pig_ge2.json": f"/mnt/D1/globus/pig_trained_{model}s/ge2/",
    #                "pig_gm4": f"/mnt/D1/globus/pig_trained_{model}s/gm4/",
    #                "pig_ge4.json": f"/mnt/D1/globus/pig_trained_{model}s/ge4/",
    #                "cov": f"/mnt/D1/globus/cov_trained_{model}s/",
    #                "ribo": f"/mnt/D1/globus/ribo_trained_{model}s/"}[datatype_str]
    #
    # # Paths from current directory (analysis) to location of data files and trained models folder
    # # data_location = {"pig_ge2.json": "../../pig/ge2/",
    # #             "pig_gm2": "../../pig/gm2/",
    # #             "pig_gm4": "../../pig/gm4/",
    # #             "pig_ge4.json": "../../pig/ge4/",
    # #             "invivo": "../../invivo/",
    # #             "rod": "../../rod/",
    # #             "cov": "../../cov/",
    # #             "ribo": "../../ribo/"}[datatype_str]
    #
    # # Server model location provides relative path from phage_display_ML directory to the server-side trained model directory
    # server_model_location = data_location[6:] + f"trained_{model}s/"
    #
    #
    #
    #
    # if datatype["focus"] == "pig":
    #     molecule = "protein"
    #     rounds = ['np1', 'np2', 'np3', 'n1', 'b3']
    #     if type(cluster) == str and cluster == "all":
    #         all_data_files, all_model_names, all_rounds, all_configkeys = [], [], [], []
    #         for i in range(datatype["clusters"]):
    #             all_data_files.append([x + f'_c{i+1}.fasta' for x in rounds])
    #             all_configkeys.append(f"{datatype_str[:4]}c{i+1}_{datatype_str[4:]}")
    #             if weights:
    #                 all_model_names.append([x + f"_c{i+1}_w" for x in rounds])
    #                 all_rounds.append([x + f"_c{i+1}" for x in rounds])
    #             else:
    #                 all_model_names.append([x + f"_c{i+1}" for x in rounds])
    #                 all_rounds.append([x + f"_c{i+1}" for x in rounds])
    #
    #         info = {"data_files": all_data_files, "rounds": all_rounds,
    #                 "model_names": all_model_names, "local_model_dir": local_model_location,
    #                 "data_dir": data_location, "server_model_dir": server_model_location,
    #                 "molecule": molecule, "cluster": "all", "configkeys": all_configkeys}
    #     else:
    #         assert cluster > 0
    #         all_data_files = [x + f'_c{cluster}.fasta' for x in rounds]
    #         if weights:
    #             all_model_names = [x + f"_c{cluster}_w" for x in rounds]
    #             all_rounds = [x + f"_c{cluster}" for x in rounds]
    #         else:
    #             all_model_names = [x + f"_c{cluster}" for x in rounds]
    #             all_rounds = all_model_names
    #
    #         info = {"data_files": all_data_files, "rounds": all_rounds,
    #                 "model_names": all_model_names, "local_model_dir": local_model_location,
    #                 "data_dir": data_location, "server_model_dir": server_model_location,
    #                 "molecule": molecule, "cluster": cluster, "configkey": f"{datatype_str[:4]}c{cluster}_{datatype_str[4:]}"}
    #
    # elif datatype["focus"] == "cov":
    #     molecule = "dna"
    #
    #     all_data_files = [f"r{i}.fasta" for i in range(1, 13)]
    #     if weights:
    #         all_rounds = [f"r{i}_w" for i in range(1, 13)]
    #     else:
    #         all_rounds = [f"r{i}" for i in range(1, 13)]
    #
    #     all_model_names = all_rounds
    #
    #     info = {"data_files": all_data_files, "rounds": all_rounds,
    #             "model_names": all_model_names, "local_model_dir": local_model_location,
    #             "data_dir": data_location, "server_model_dir": server_model_location,
    #             "molecule": molecule, "configkey": f"{datatype_str[:3]}"}
    #
    # elif datatype["focus"] == "invivo":
    #     molecule = "protein"
    #     # dest_path = f"../invivo/trained_{model}s/"
    #     # src_path = "../invivo/"
    #     #
    #     # c1 = [x + '_c1.fasta' for x in rounds]
    #     # c2 = [x + '_c2.fasta' for x in rounds]
    #     #
    #     # all_data_files = c1 + c2
    #     #
    #     # model1 = [x + '_c1' for x in rounds]
    #     # model2 = [x + '_c2' for x in rounds]
    #     #
    #     # all_model_names = model1 + model2
    #     #
    #     # script_names = ["invivo" + str(i) for i in range(len(all_model_names))]
    #     #
    #     # paths_to_data = [src_path + x for x in all_data_files]
    #
    # elif datatype["focus"] == "ribo":
    #
    #
    #
    # return info