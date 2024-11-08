from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os
from .schema import ResultInfoModel
SERVICE_ACCOUNT_FILE = os.path.join(os.getcwd(), 'service_account.json')

def upload_to_drive(folder_name: str, full_file_name: str,SendInfo:ResultInfoModel):
    # Path to your service account key file
    
    
    # Ensure the file exists
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(f"Service account file not found at {SERVICE_ACCOUNT_FILE}")
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    # Authenticate using the service account
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)

    # Function to get or create a folder in Google Drive
    def get_or_create_folder(drive_service, folder_name, parent_id=None):
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]

            folder = drive_service.files().create(body=file_metadata, fields='id').execute()
            return folder.get('id')
        else:
            return items[0]['id']

    # Use the function to get or create the folder
    folder_id = get_or_create_folder(drive_service, folder_name, '1ey0C4FkqFHIePfY9Lu0XzXCAOdokbMc3')

    # File metadata and media to be uploaded
    file_metadata = {
        'name': full_file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(full_file_name, mimetype='text/csv')

    # Upload the file
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    SendInfo.FileId= file.get('id')
    print('File ID: %s' % file.get('id'))

    # Remove the local files after upload
    os.remove(full_file_name)
    print("Local files removed")
    return

