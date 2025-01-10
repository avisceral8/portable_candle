# Tableau User Management Script 🚀

A Python script that takes the tedium out of managing Tableau users. This script automates the process of adding users to your Tableau server AND assigning them to their respective groups, all from a simple CSV file.

## What Does This Script Do? 

This script is your friendly automation helper that:
1. Reads user information from a CSV file
2. Connects to your Tableau server using personal access tokens
3. Creates new user accounts
4. Assigns users to their designated groups
5. Keeps track of any users that couldn't be added

Think of it as your virtual HR assistant for Tableau user management!

## Prerequisites

Before you can run this script, you'll need:

- Python 3.x installed on your system
- The following Python packages:
  - pandas
  - tableauserverclient
- A Tableau Server instance with administrative access
- A personal access token for authentication
- A CSV file containing user information
- All groups mentioned in your CSV file must be created on your Tableau server before running the script. The script can assign users to existing groups, but it cannot create new groups. 

## Setting Up Your CSV File

Your CSV file should include the following columns:
- Username: The user's full name
- Email: The user's email address (this will be their unique identifier)
- Site Role: The user's role in Tableau
- Group: The group they should be assigned to

## Configuration

In the script, you'll need to update these variables with your specific information:

```python
server_address = "Insert site address here"
site_name = "Insert site name here"
token_name = "Insert token name here"
token_value = "Insert secret token here"
```

## Running the Script

1. Place your Users.csv file in the same directory as the script
2. Update the configuration variables
3. Run the script: `python CreateUsers.py`

The script will provide real-time feedback as it processes each user, and it will create a "Users_not_added.csv" file if any users couldn't be added successfully.

## Error Handling

If something goes wrong while adding a user, don't worry! The script:
- Continues processing the remaining users
- Records the error details
- Saves all unsuccessful attempts to "Users_not_added.csv"

## Performance

The script includes a timer to track how long the process takes. This is particularly useful when adding large numbers of users, helping you plan future bulk user additions.

## Need Help?

If you encounter any issues:
1. Check that your CSV file is formatted correctly
2. Verify your server address and token credentials
3. Ensure you have the necessary permissions on your Tableau server
4. Review the "Users_not_added.csv" file for specific error messages
