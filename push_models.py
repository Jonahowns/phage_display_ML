import globus_sdk
from glob import glob
import argparse
import rbm_torch.global_info

# Should be run on Agave, Pushes latest models to our Work Dell
# Goal is to Transfer the most recent version of each RBM automatically to our local computer for analysis


# find the latest version and return the path source_dir/version_{max}/
# Endpoint dir is for the endpoint being mounted in a different place then where the script is run
# For example, all paths on agave have "/scratch/" as their first directory
# So to use the agave scratch endpoint to transfer files we need to take into account that scratch is mounted directly at /scratch
# Any paths with /scratch in them will fail
def find_version(round, source_dir, endpoint_dir=None):
    path = source_dir+round
    subdirs = glob(path+"/*/", recursive=True)  # list of all versions of the RBM
    versions = [int(x[:-1].rsplit("_")[-1]) for x in subdirs]  # extracted version numbers
    try:
        maxv = max(versions)  # get highest version number
    except ValueError:
        print(f"No Model {round} found in {path}")
        exit(-1)
    indexofinterest = versions.index(maxv)  # Get index of the highest version
    targetdir = subdirs[indexofinterest]  # Access directory path of the highest version

    if endpoint_dir:
        last_two_dirs = "/".join(targetdir.split("/")[-3:])  # Get the last two directories
        fullpath = endpoint_dir+last_two_dirs  # Make path relative to our endpoint
        return fullpath
    else:
        return targetdir

# Create Transfer Task and Submit
def model_transfer(rounds, source_dir, source_endpoint_dir, dest_dir):
    # create a Transfer task consisting of one or more items
    task_data = globus_sdk.TransferData(
        transfer_client, source_endpoint_id, dest_endpoint_id
    )

    for x in rounds:
        # Find the
        version_dir = find_version(x, source_dir, endpoint_dir=source_endpoint_dir)
        # Destination of our Trained Models RIP
        final_destination = dest_dir + x + "/" + version_dir.split("/")[-2] + "/"

        task_data.add_item(
            version_dir,  # source
            final_destination,  # dest
            recursive=True  # Transfer Directory Contents
        )

    # submit, getting back the task ID
    task_doc = transfer_client.submit_transfer(task_data)
    task_id = task_doc["task_id"]
    print(f"submitted transfer, task_id={task_id}")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Transfer trained rbm or crbm models from Agave Scratch Endpoint to Target Endpoint")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('ml_model', type=str, help="Which model? rbm or crbm?")
    requiredNamed.add_argument('datatype_str', type=str, help="Which dataset to transfer")
    requiredNamed.add_argument('model_string', type=str, nargs="+", help="Which models to transfer, c1")
    args = parser.parse_args()

    info = rbm_torch.global_info.get_global_info(args.datatype_str, dir="./datasets/dataset_files/")

    destination_dir = info["local_model_dir"][args.ml_model]

    ###################################################################################################
    ####### Change what's within the pound sign lines

    source_dir = f"/scratch/jprocyk/machine_learning/phage_display_ML/{info['server_model_dir'][args.ml_model]}"

    source_endpoint_dir = f"/jprocyk/machine_learning/phage_display_ML/{info['server_model_dir'][args.ml_model]}"

    # Globus Transfer Information
    # Configured "App" for Globus Transfers
    app_client_id = "fc83d632-e246-4a03-99ed-78e18f215bd1"

    # Agave Scratch Endpoint
    source_endpoint_id = "e42349b0-ceff-44a5-bfb6-c0b5a19c32c7"

    # Work Dell Endpoint (my globus personal endpoint)
    dest_endpoint_id = "493bcdda-f6fd-11eb-b46a-eb47ba14b5cc"

    # Additional data access scope Needed by Agave Scratch endpoint since it is a V5 globus endpoint
    source_scope = f"urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/{source_endpoint_id}/data_access]"

    ########################################################################################################

    # MAKE MORE GENERAL
    clusters = info["clusters"] # number of clusters
    rounds = []
    for clust in range(clusters):
        rounds.append(info["model_names"][str(clust+1)])

    flat_rounds = [item for sublist in rounds for item in sublist]

    individual = []  # individual round specifiers are added here for one call to model_transfer at the end
    for specifier in args.model_string:  # specifier is the rbm name, ususally something like n1_c2_w etc.
        if "all" in specifier:  # Transfer all rbms under this datatype_str
            if specifier == "all":
                individual += flat_rounds
            else:
                extension = specifier.split('_')[1]
                individual += [f"{x}_{extension}" for x in flat_rounds]
        else:
            individual.append(specifier)


    # Now we setup the Globus transfer
    client = globus_sdk.NativeAppAuthClient(app_client_id)
    # Authorization
    client.oauth2_start_flow(refresh_tokens=True, requested_scopes=[source_scope])
    authorize_url = client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")
    auth_code = input("Please enter the code you get after login here: ").strip()
    tokens = client.oauth2_exchange_code_for_tokens(auth_code)

    # Get Transfer Tokens for our transfer
    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

    # construct an AccessTokenAuthorizer and use it to construct the TransferClient
    transfer_client = globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    )

    model_transfer(individual, source_dir, source_endpoint_dir, destination_dir)

