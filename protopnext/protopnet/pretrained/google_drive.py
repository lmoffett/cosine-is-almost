import fcntl
import logging
import time
from pathlib import Path

import requests
import tqdm
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def parse_gdrive_download_form(html_content):
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the form with id "download-form"
    form = soup.find("form", {"id": "download-form"})

    if not form:
        raise ValueError("Form with id 'download-form' not found in the HTML content.")

    # Extract all hidden input fields except for the submit button
    hidden_inputs = form.find_all("input", {"type": "hidden"})

    # Build a dictionary of name-value pairs for the hidden inputs
    form_data = {}
    for hidden_input in hidden_inputs:
        name = hidden_input.get("name")
        value = hidden_input.get("value", "")
        if name:  # Ensure the input has a name attribute
            form_data[name] = value

    return form_data


def download_from_google_drive(file_id: str, destination: Path):
    """
    Download a large file from Google Drive using the file ID.
    The file is saved to the destination path.

    Args:
        file_id (str): The ID of the file on Google Drive.
        destination (Path): The destination path to save the file.
    """
    # Base URL for downloading files from Google Drive
    url = "https://drive.google.com/uc?export=download"

    # Start a session
    session = requests.Session()

    # Initial request to get the confirmation token
    response = session.get(url, params={"id": file_id}, stream=True)

    download_form_data = parse_gdrive_download_form(response.content)

    # Make the download request with the confirm token and UUID
    log.info("Downloading google drive file %s to %s", file_id, destination)
    model_response = session.get(
        "https://drive.usercontent.google.com/download",
        params={
            key: download_form_data[key] for key in ["id", "confirm", "export", "uuid"]
        },
        stream=True,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "wb") as f:
        for chunk in tqdm.tqdm(model_response.iter_content(chunk_size=32768)):
            if chunk:  # Filter out keep-alive new chunks
                f.write(chunk)

    log.debug("File downloaded to %s", destination)


def ensure_pretrained_model_from_google_drive(
    file_id: str, destination: Path, timeout: int = 30
):
    """
    Ensures the pretrained model is downloaded to the destination, using optimistic locking.

    Args:
        file_id (str): The Google Drive file ID.
        destination (Path): The path to download the file to.
        timeout (int): Maximum time (in seconds) to wait for a lock before giving up.

    Raises:
        TimeoutError: If unable to acquire the lock within the timeout.
        Exception: If the download fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(".lock")
    start_time = time.time()

    # Attempt to acquire lock
    with open(lock_path, "w") as lock_file:
        while True:
            try:
                # Attempt to lock the file
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired, proceed
            except BlockingIOError:
                log.debug("Waiting for lock on %s", destination)
                # Check if timeout has been reached
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock for {destination} within {timeout} seconds."
                    )
                time.sleep(1)  # Wait before retrying

        # Lock acquired; check if file already exists
        if destination.exists():
            log.debug("Using cached file: %s", destination)
            return

        # File does not exist; proceed to download
        try:
            download_from_google_drive(file_id, destination)
        except Exception as e:
            log.error(f"Could not acquire file from Google Drive: {e}")
            raise
        finally:
            # Release the lock
            fcntl.flock(lock_file, fcntl.LOCK_UN)
