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
cov_sw_datatype = {"focus": "cov", "molecule": "dna", "id": "sw", "process": "scaled_weights", "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[40, 40]]}
cov_nw_datatype = {"focus": "cov", "molecule": "dna", "id": "nw", "process": "scaled_negative_weights", "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[40, 40]]}
ribo_datatype = {"focus": "ribo", "molecule": "rna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[115, 121]]}
thc_datatype = {"focus": "thc", "molecule": "rna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[41, 43]]}
pal_datatype = {"focus": "pal", "molecule": "dna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[39, 41]]}
exo_datatype = {"focus": "exo", "molecule": "dna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[35, 38]]}
exo_sw_datatype = {"focus": "exo", "molecule": "dna", "id": "sw", "process": "scaled_weights", "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[35, 38]]}

# datatype_str : very important string specifier. For clustered datasets set as datatype["focus"]_datatype["id"].
# For non-clustered datasets with no id set as datatype["focus"]
# For non-clustered datasets with an id, it is set as datatype["focus"]_datatype["id"]
datatype_list = [
    pig_ge2_datatype,
    pig_gm2_datatype,
    pig_ge4_datatype,
    pig_gm4_datatype,
    cov_datatype,
    cov_sw_datatype,
    cov_nw_datatype,
    ribo_datatype,
    thc_datatype,
    pal_datatype,
    exo_datatype,
    exo_sw_datatype
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


class DatasetInfo:
    def __init__(self):
        self.info = None

    def load_dataset(self, datatype_str, dir="../../datasets/dataset_files/"):
        with open(dir + datatype_str + ".json", "r") as json_file:
            info = json.load(json_file)
        self.info = info

    def generate_dataset(self, datatype_str, destination="../../datasets/dataset_files/"):
        generate_dataset_file(self.info["data_files"], supported_datatypes[datatype_str], destination=destination)

    def modify_value(self, key, new_value):
        self.info[key] = new_value

    def get_value(self, key):
        return self.info[key]

    # most common operation
    def add_data_file(self, dataset_file_basename):
        new_data_files = self.get_value("data_files")
        model_names = self.get_value("model_names")
        rounds = self.get_value("rounds")

        # config key doesn't change here, maybe add that for more flexibility later?
        clusters = self.get_value("clusters") # get number of clusters
        if clusters > 1:
            for cluster in range(1, clusters+1):
                # append data file to previous list
                new_data_files[str(cluster)].append(f'{dataset_file_basename}_c{cluster}.fasta')
                model_names[str(cluster)].append(f'{dataset_file_basename}_c{cluster}')
                rounds[str(cluster)].append(f"{dataset_file_basename}_c{cluster}")
        else:
            cluster = "1"
            new_data_files[cluster].append(f'{dataset_file_basename}.fasta')
            model_names[cluster].append(dataset_file_basename)
            rounds[cluster].append(dataset_file_basename)

        self.modify_value("data_files", new_data_files)
        self.modify_value("model_names", model_names)
        self.modify_value("rounds", rounds)


def load_dataset(datatype_str, dir="../../datasets/dataset_files/"):
    with open(dir+datatype_str+".json", "r") as json_file:
        info = json.load(json_file)
    return info


def generate_dataset_file(data_filenames, datatype, destination="../../datasets/dataset_files/"):
    # rounds are assigned by each filename
    rounds = [x.split(".")[0] for x in data_filenames]

    assert type(datatype) is dict

    if datatype["id"] is not None:
        data_location = f"../../datasets/{datatype['focus']}/{datatype['id']}/"
    else:
        data_location = f"../../datasets/{datatype['focus']}/"

    server_model_location, local_model_location = {}, {}
    for model in supported_ml_models:
        # Server model location provides relative path from phage_display_ML directory to the server-side trained model directory
        server_model_location[model] = data_location[6:] + f"trained_{model}s/"
        # Paths from current directory (analysis) to location of data files and trained models folder
        if datatype["id"] is not None:
            local_model_location[model] = f"/mnt/D1/globus/{datatype['focus']}_trained_{model}s/{datatype['id']}/"
        else:
            local_model_location[model] = f"/mnt/D1/globus/{datatype['focus']}_trained_{model}s/"
    # dictionaries with cluster number as the key
    all_data_files, all_model_names, all_model_names_w, all_rounds, all_configkeys = {}, {}, {}, {}, {}

    if datatype["clusters"] > 1:
        for i in range(datatype["clusters"]):
            all_data_files[i+1] = [x + f'_c{i + 1}.fasta' for x in rounds]

            all_configkeys[i+1] = f"{datatype['focus']}_c{i + 1}_{datatype['id']}"

            all_model_names[i+1] = [x + f"_c{i + 1}" for x in rounds]
            # all_model_names_w[i+1] = [x + f"_c{i + 1}_w" for x in rounds]

            all_rounds[i+1] = [x + f"_c{i + 1}" for x in rounds]

    else:
        all_data_files[1] = [f"{i}.fasta" for i in rounds]

        all_rounds[1] = rounds

        all_model_names[1] = [f"{i}" for i in rounds]
        # all_model_names_w[1] = [f"{i}_w" for i in rounds]

        if datatype["id"] is None:
            all_configkeys[1] = f"{datatype['focus']}"
        else:
            all_configkeys[1] = f"{datatype['focus']}_{datatype['id']}"

    info = {"data_files": all_data_files, "rounds": all_rounds,
            "model_names": all_model_names,
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
