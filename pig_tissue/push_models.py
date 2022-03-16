import globus_sdk
from glob import glob
import argparse

# Should be run on Agave, Pushes latest models to our Work Dell
# Goal is to Transfer the most recent version of each RBM automatically to our local computer for analysis

# Configured "App" for Globus Transfers
piggy_client_id = "fc83d632-e246-4a03-99ed-78e18f215bd1"

# Agave Scratch Endpoint
source_endpoint_id = "e42349b0-ceff-44a5-bfb6-c0b5a19c32c7"

# Work Dell Endpoint
dest_endpoint_id = "493bcdda-f6fd-11eb-b46a-eb47ba14b5cc"

# Additional data access scope Needed by Agave Scratch endpoint since it is a V5 globus endpoint
source_scope = f"urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/{source_endpoint_id}/data_access]"


client = globus_sdk.NativeAppAuthClient(piggy_client_id)
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


## Let's Get Our Specifics, Source of the trained RBMS
source_dir = "/scratch/jprocyk/machine_learning/phage_display_ML/pig_tissue/trained_rbms/"
# Path from our endpoint since it's mounted at /scratch
source_endpoint_dir = "/jprocyk/machine_learning/phage_display_ML/pig_tissue/trained_rbms/"

# local destination on my computer
dest_dir = "/mnt/D1/globus/pig_trained_rbms/"


# find the latest version and return the path source_dir/version_{max}/
# Endpoint dir is for the endpoint being mounted in a different place then where the script is run
# For example, all paths on agave have "/scratch/" as their first directory
# So to use the agave scratch endpoint to transfer files we need to take into account that scratch is mounted directly at /scratch
# Any paths with /scratch in them will fail
def find_version(round, dir=source_dir, endpoint_dir=None):
    path = dir+round
    subdirs = glob(path+"/*/", recursive=True)  # list of all versions of the RBM
    versions = [int(x[:-1].rsplit("_")[-1]) for x in subdirs]  # extracted version numbers
    maxv = max(versions)  # get highest version number
    indexofinterest = versions.index(maxv)  # Get index of the highest version
    targetdir = subdirs[indexofinterest]  # Access directory path of the highest version

    if endpoint_dir:
        last_two_dirs = "/".join(targetdir.split("/")[-3:])  # Get the last two directories
        fullpath = endpoint_dir+last_two_dirs  # Make path relative to our endpoint
        return fullpath
    else:
        return targetdir

# Create Transfer Task and Submit
def rbm_transfer(rounds, source=source_dir, dest=dest_dir):
    # create a Transfer task consisting of one or more items
    task_data = globus_sdk.TransferData(
        transfer_client, source_endpoint_id, dest_endpoint_id
    )

    for x in rounds:
        # Find the
        version_dir = find_version(x, dir=source, endpoint_dir=source_endpoint_dir)
        # Destination of our Trained RBMS
        final_destination = dest + x + "/" + version_dir.split("/")[-2] + "/"

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
    parser = argparse.ArgumentParser(description="RBM Training on Phage Display Dataset")
    parser.add_argument('model_string', type=str, nargs="+", help="Which models to transfer, c1")
    args = parser.parse_args()

    # The RBMS string specifiers
    rounds = ["b3", "n1", "np1", "np2", "np3"]
    c1_rounds = [x + "_c1" for x in rounds]
    c1_rounds_w = [x + "_c1_w" for x in rounds]
    c2_rounds = [x + "_c2" for x in rounds]
    c2_rounds_w = [x + "_c2_w" for x in rounds]
    all_rounds = c1_rounds+c1_rounds_w+c2_rounds+c2_rounds_w
    

    individual = [] # individual round specifiers are added here for one call to rbm_transfer at the end
    for specifier in args.model_string:
        rbm_transfer(c1_rounds) if specifier == "c1" else None
        rbm_transfer(c2_rounds) if specifier == "c2" else None
        rbm_transfer(c1_rounds_w) if specifier == "c1_w" else None
        rbm_transfer(c2_rounds_w) if specifier == "c2_w" else None
        rbm_transfer(all_rounds) if specifier == "all" else None
        if specifier in all_rounds:
            individual.append(specifier)

    if individual:
        rbm_transfer(individual)


