def get_global_info(datatype_str, cluster=0, weights=False):
    pig_ge2_datatype = {"focus": "pig", "molecule": "protein", "id": "ge2", "process": "gaps_end", "clusters": 2, "gap_position_indices": [-1, -1], "cluster_indices": [[12, 22], [35, 45]]}
    pig_gm2_datatype = {"focus": "pig", "molecule": "protein", "id": "gm2", "process": "gaps_middle", "clusters": 2, "gap_position_indices": [2, 16], "cluster_indices": [[12, 22], [35, 45]]}  # based solely off looking at the sequence logos
    pig_ge4_datatype = {"focus": "pig", "molecule": "protein", "id": "ge4", "process": "gaps_end", "clusters": 4, "gap_position_indices": [-1, -1, -1, -1], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}
    pig_gm4_datatype = {"focus": "pig", "molecule": "protein", "id": "gm4", "process": "gaps_middle", "clusters": 4, "gap_position_indices": [2, 2, 16, 16], "cluster_indices": [[12, 16], [17, 22], [35, 39], [40, 45]]}

    cov_datatype = {"focus": "cov", "molecule": "dna", "id": None, "process": None, "clusters": 1, "gap_position_indices": [-1], "cluster_indices": [[40, 40]]}

    try:
        datatype = {"pig_ge2": pig_ge2_datatype,
                    "pig_gm2": pig_gm2_datatype,
                    "pig_ge4": pig_ge4_datatype,
                    "pig_gm4": pig_gm4_datatype,
                    "cov": cov_datatype}[datatype_str]
    except KeyError:
        raise KeyError


    local_rbm_location = {"pig_gm2": "/mnt/D1/globus/pig_trained_rbms/gm2/",
                   "pig_ge2": "/mnt/D1/globus/pig_trained_rbms/ge2/",
                   "pig_gm4": "/mnt/D1/globus/pig_trained_rbms/gm4/",
                   "pig_ge4": "/mnt/D1/globus/pig_trained_rbms/ge4/",
                   "cov": "/mnt/D1/globus/cov_trained_rbms/"}[datatype_str]

    # Paths from current directory (analysis) to location of data files and trained rbms folder
    data_location = {"pig_ge2": "../../pig_tissue/gaps_end_2_clusters/",
                "pig_gm2": "../../pig_tissue/gaps_middle_2_clusters/",
                "pig_gm4": "../../pig_tissue/gaps_middle_4_clusters/",
                "pig_ge4": "../../pig_tissue/gaps_end_4_clusters/",
                "invivo": "../../invivo/",
                "rod": "../../rod/",
                "cov": "../../cov/"}[datatype_str]

    # Server rbm location provides relative path from phage_display_ML directory to the co
    server_rbm_location = data_location[6:] + "trained_rbms/"

    assert datatype["focus"] in ["pig", "cov"]

    if datatype["focus"] == "pig":
        molecule = "protein"
        rounds = ['np1', 'np2', 'np3', 'n1', 'b3']
        if type(cluster) == str and cluster == "all":
            all_data_files, all_rbm_names, all_rounds, all_configkeys = [], [], [], []
            for i in range(datatype["clusters"]):
                all_data_files.append([x + f'_c{i+1}.fasta' for x in rounds])
                all_configkeys.append(f"{datatype_str[:4]}c{i+1}_{datatype_str[4:]}")
                if weights:
                    all_rbm_names.append([x + f"_c{i+1}_w" for x in rounds])
                    all_rounds.append([x + f"_c{i+1}" for x in rounds])
                else:
                    all_rbm_names.append([x + f"_c{i+1}" for x in rounds])
                    all_rounds.append([x + f"_c{i+1}" for x in rounds])

            info = {"data_files": all_data_files, "rounds": all_rounds,
                    "rbm_names": all_rbm_names, "local_rbm_dir": local_rbm_location,
                    "data_dir": data_location, "server_rbm_dir": server_rbm_location,
                    "molecule": molecule, "cluster": "all", "configkeys": all_configkeys}
        else:
            assert cluster > 0
            all_data_files = [x + f'_c{cluster}.fasta' for x in rounds]
            if weights:
                all_rbm_names = [x + f"_c{cluster}_w" for x in rounds]
                all_rounds = [x + f"_c{cluster}" for x in rounds]
            else:
                all_rbm_names = [x + f"_c{cluster}" for x in rounds]
                all_rounds = all_rbm_names

            info = {"data_files": all_data_files, "rounds": all_rounds,
                    "rbm_names": all_rbm_names, "local_rbm_dir": local_rbm_location,
                    "data_dir": data_location, "server_rbm_dir": server_rbm_location,
                    "molecule": molecule, "cluster": cluster, "configkey": f"{datatype_str[:4]}c{cluster}_{datatype_str[4:]}"}

    elif datatype["focus"] == "cov":
        molecule = "dna"

        all_data_files = [f"r{i}.fasta" for i in range(1, 13)]
        if weights:
            all_rounds = [f"r{i}_w" for i in range(1, 13)]
        else:
            all_rounds = [f"r{i}" for i in range(1, 13)]

        all_rbm_names = all_rounds

        info = {"data_files": all_data_files, "rounds": all_rounds,
                "rbm_names": all_rbm_names, "local_rbm_dir": local_rbm_location,
                "data_dir": data_location, "server_rbm_dir": server_rbm_location,
                "molecule": molecule, "configkey": f"{datatype_str[:3]}"}

    return info