#####################################################################################################
# Python Module for to load data into UiPath API - Coinbase Demo                                    #
# Author: Subhadip Kundu (Jade Global)                                                              #
# --------------------------------------------------------------------------------------------------#
#    Date      |     Author          |                   Comment                                    #
# ------------ + ------------------- + ------------------------------------------------------------ #
# 15-Jun-2024  | Subhadip Kundu      | Created the Initial Code                                     #
#####################################################################################################

import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

# Declare Global Variables
UiPath_Account_Name = os.getenv("UiPath_Account_Name")
UiPath_Tenant_Name = os.getenv("UiPath_Tenant_Name")
UiPath_Queue_Name = os.getenv("UiPath_Queue_Name")
UiPath_Client_Id = os.getenv("UiPath_Client_Id")
UiPath_User_Key = os.getenv("UiPath_User_Key")
UiPath_File_Id = os.getenv("UiPath_File_Id")
orchestrator_url = 'https://cloud.uipath.com'
queue_item_id = ""
oauth_token = ""

def get_oauth_token():
    auth_url = f'{orchestrator_url}/identity_/connect/token'
    auth_headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    auth_data = {
        'grant_type': 'client_credentials',
        'client_id': UiPath_Client_Id,
        'client_secret': UiPath_User_Key,
        'scope': 'OR.Queues OR.Folders OR.Folders.Read OR.Folders.Write OR.Queues.Read OR.Queues.Write'
    }
    response = requests.post(auth_url, headers=auth_headers, data=auth_data)
    if response.status_code == 200:
        token = response.json()['access_token']
        return token
    else:
        raise Exception(f"Failed to get OAuth token: {response.text}")

def add_data_to_queue(api_command):
    add_queue_item_url = f'{orchestrator_url}/{UiPath_Account_Name}/{UiPath_Tenant_Name}/odata/Queues/UiPathODataSvc.AddQueueItem'
    global oauth_token
    oauth_token = get_oauth_token()
    add_queue_item_headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {oauth_token}',
        'X-UIPATH-OrganizationUnitId': f'{UiPath_File_Id}'
    }

    queue_data = {
        "itemData": {
            "Priority": "Normal",
            "Name": UiPath_Queue_Name,
            "SpecificContent": {
                "ApiCommand@odata.type": "#String",
                "ApiCommand": api_command
            }
        }
    }

    post_response = requests.post(add_queue_item_url, headers=add_queue_item_headers, data=json.dumps(queue_data))
    global queue_item_id
    queue_item_id = post_response.json()["Id"]

    if post_response.status_code == 201:
        return "Queue item added successfully."
    else:
        return f"Failed to add queue item: {post_response.status_code} {post_response.text}"

def read_status_in_queue():
    try:
        get_queue_item_url = f'{orchestrator_url}/{UiPath_Account_Name}/{UiPath_Tenant_Name}/orchestrator_/odata/QueueItems({queue_item_id})'
        get_queue_item_headers = {
            'Authorization': f'Bearer {oauth_token}',
            'X-UIPATH-OrganizationUnitId': f'{UiPath_File_Id}'
        }
        get_response = requests.get(get_queue_item_url, headers=get_queue_item_headers)
        if get_response.status_code == 200:
            queue_item_response = get_response.json()
            queue_item_status = queue_item_response['Status']
            queue_item_progress = queue_item_response['Progress']
            return queue_item_status, queue_item_progress
        else:
            return "Failed to connect to the queue:", f"{get_response.status_code} {get_response.text}"

    except Exception as err:
        return "Failed to get queue item status:", f"{err}"
