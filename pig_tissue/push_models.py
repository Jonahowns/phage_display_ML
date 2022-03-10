import globus_sdk
from glob import glob

# Should be run on Agave, Pushes latest models to our Work Dell

# Goal is to Transfer the most recent version of each RBM automatically to our local computer for analysis

# Configured "App" for Globus Transfers
piggy_client_id = "fc83d632-e246-4a03-99ed-78e18f215bd1"

client = globus_sdk.NativeAppAuthClient(piggy_client_id)
# Authorization
client.oauth2_start_flow()
authorize_url = client.oauth2_get_authorize_url()
print(f"Please go to this URL and login:\n\n{authorize_url}\n")
auth_code = input("Please enter the code you get after login here: ").strip()
tokens = client.oauth2_exchange_code_for_tokens(auth_code)
# Get Transfer Tokens for our transfer
transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

# construct an AccessTokenAuthorizer and use it to construct the
# TransferClient
transfer_client = globus_sdk.TransferClient(
    authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
)

# Agave Scratch Endpoint
source_endpoint_id = "e42349b0-ceff-44a5-bfb6-c0b5a19c32c7"
# Work Dell Endpoint
dest_endpoint_id = "493bcdda-f6fd-11eb-b46a-eb47ba14b5cc"


## Let's Get Our Specifics, Source of the trained RBMS
source_dir = "/jprocyk/machine_learning/phage_display_ML/pig_tissue/trained_rbms/"
dest_dir = "/mnt/D1/globus/pig_trained_rbms/"
# The RBMS in question
rounds = ["b3", "n1", "np1", "np2", "np3"]
c1_rounds = [x+"_c1" for x in rounds]
c2_rounds = [x+"_c2" for x in rounds]


# find the latest version and return the path source_dir/version_{max}/
def find_version(round, dir=source_dir):
    path = dir+round
    subdirs = glob(path+"/*/", recursive=True)  # list of all versions of the RBM
    versions = [int(x[:-1].rsplit("_")[-1]) for x in subdirs]  # extracted version numbers
    maxv = max(versions)  # get highest version number
    indexofinterest = versions.index(maxv)  # Get index of the highest version
    targetdir = subdirs[indexofinterest]  # Access directory path of the highest version
    return targetdir

# Create Transfer Task and Submit
def rbm_transfer(rounds, source=source_dir, dest=dest_dir):
    # create a Transfer task consisting of one or more items
    task_data = globus_sdk.TransferData(
        transfer_client, source_endpoint_id, dest_endpoint_id
    )

    for x in rounds:
        version_dir = find_version(x, dir=source)
        # Destination of our Trained RBMS
        final_destination = dest + x + "/"

        task_data.add_item(
            version_dir,  # source
            final_destination,  # dest
            recursive=True  # Transfer Whole Directory
        )

    # submit, getting back the task ID
    task_doc = transfer_client.submit_transfer(task_data)
    task_id = task_doc["task_id"]
    print(f"submitted transfer, task_id={task_id}")


# Transfer all Models
rbm_transfer(rounds)

# Transfer all c1
# rbm_transfer(c1_rounds)

# Transfer Specific
# rs = ["b3_c1", "n1_c1"]
# rbm_transfer(rs)



