#Loading Dependencies
import pandas as pd
import tableauserverclient as TSC
import time

print("\nDependencies Loaded")
start_time = time.perf_counter()

#Input variables for Token
server_address = "Insert site address here"
site_name = "Insert site name here"
token_name = "Insert token name here"
token_value = "Insert secret token here"

df = pd.read_csv("Users.csv")
print("\nUsers File Loaded")

#Accessing the server
tableau_auth = TSC.PersonalAccessTokenAuth(
    token_name=token_name,
    personal_access_token=token_value,
    site_id=server_address,
    )

server = TSC.Server(server_address,use_server_version=True)
print("\nAuthentication Successful")

#Priming empty variable to collect users which didnt get added
user_not_added = []

#Main body which would read through each row, add a user and add them to a group
with server.auth.sign_in(tableau_auth):

    #Looping through each row
    for index,row in df.iterrows():
        fullname = row['Username']
        id = row['Email']
        site_role = row['Site Role']
        groupname = row['Group']

        group_id = user_id = ""

        try:
            new_user = TSC.UserItem(id,site_role)
            new_user = server.users.add(new_user)

            #Searching for User ID
            all_users = list(TSC.Pager(server.users))
            for user in all_users:
                if user.name == id:
                    user_id = user.id

            #Searching for Group ID
            all_groups = list(TSC.Pager(server.groups))
            for group in all_groups:
                if group.name == groupname:
                    group_id = group
            
            #Adding user to group
            server.groups.add_user(group_id,user_id)
            print(f"\n Added {user.name} to {groupname}")
        
        except Exception as e:
            user_not_added.append({"User_ID":id,"Error":str(e)})
            print(f"\nAn error occured while adding user {id}")
            continue

#Save list of users not added to a csv
user_not_added = pd.DataFrame(user_not_added)
user_not_added.to_csv("Users_not_added.csv", index=False)

end_time =  time.perf_counter()
runtime = (end_time-start_time)/60
print(f"\nRuntime is {runtime:.2f} minutes")