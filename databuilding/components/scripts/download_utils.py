import time
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import os
import json
from colorama import Fore, Back, Style
import numpy as np
from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners


from pathlib import Path

# Get the current file's directory (where the script is running or located)
current_file_directory = Path(__file__).resolve().parent


# Find the root folder by traversing upwards
def find_root_folder(current_path, root_folder_name="LLM_Thesis"):
    for parent in current_path.parents:
        if parent.name == root_folder_name:
            return parent
    return None  # Return None if the folder is not found


root_folder = find_root_folder(current_file_directory)

print(f"Root folder: {root_folder}")


def download_file_url(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    with open(output_path, "wb") as file, tqdm(
        desc=output_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)


def download_all_json_files_from_url_norm(base_url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a")

    for link in links:
        file_url = link.get("href")
        if file_url.endswith(".json"):
            full_url = os.path.join(base_url, file_url)
            output_path = os.path.join(output_dir, file_url)
            download_file_url(full_url, output_path)


def remove_duplicates_from_dict(data_dict):
    for key in data_dict:
        seen = set()
        unique_list = []
        for item in data_dict[key]:
            # Create a tuple of the relevant parameters
            entity_tuple = (
                tuple(item["pos"]),
                item["type"],
                item["sent_id"],
                item["name"],
            )
            if entity_tuple not in seen:
                seen.add(entity_tuple)
                unique_list.append(item)
        data_dict[key] = unique_list
    return data_dict


def fix_grouped_entities(combined_df):
    # Assuming combined_df is already defined and loaded with data
    new_nyNER = np.empty(len(combined_df), dtype=object).tolist()
    new_relations = np.empty(len(combined_df), dtype=object).tolist()
    # Process each row
    for i in tqdm(
        range(len(combined_df)),
        total=len(combined_df),
        desc="Processing rows - Fixing grouped entities",
    ):
        if (
            combined_df.iloc[i]["org_dataset"] == "DocRED"
            or combined_df.iloc[i]["org_dataset"] == "ReDocRED_Joint"
        ):
            row = combined_df.iloc[i]
            # Initialize a dictionary to store the NER data for the current row
            NER_DICT = {}

            # Iterate through each NER entry in the row's 'NER' column
            for ner in row["NER"]:
                # Create a unique key for each name, type combination
                key = (ner["name"], ner["type"])

                # Append the entity to the list for this key in NER_DICT
                if key not in NER_DICT:
                    NER_DICT[key] = [ner]
                else:
                    NER_DICT[key].append(ner)

            # Update the 'nyNER' column in the DataFrame with the populated dictionary values
            new_nyNER[i] = list(NER_DICT.values())
            new_relations[i] = combined_df.iloc[i][
                "labels"
            ]  # Append original relations for 'DocRED'
        if combined_df.iloc[i]["org_dataset"] != "DocRED":
            relations = combined_df.iloc[i]["labels"]
            NER = combined_df.iloc[i]["vertexSet"]
            nyNER = {}  # Initialize the dictionary for this row
            counter = 0
            tmp_dickies = {}

            for jdx, rel in enumerate(relations):
                e1 = NER[rel["h"]]
                e2 = NER[rel["t"]]

                # Check if the tuple (e1 name, e1 type) exists as a key, if not, initialize as empty list
                if (e1["name"], e1["type"]) not in nyNER:
                    nyNER[(e1["name"], e1["type"])] = []
                    nyNER[(e1["name"], e1["type"])].append(e1)
                    tmp_dickies[(e1["name"], e1["type"])] = counter
                    counter += 1

                if (e2["name"], e2["type"]) not in nyNER:
                    nyNER[(e2["name"], e2["type"])] = []
                    nyNER[(e2["name"], e2["type"])].append(e2)
                    tmp_dickies[(e2["name"], e2["type"])] = counter
                    counter += 1

                # Update indices for relations
                relations[jdx]["h"] = tmp_dickies[(e1["name"], e1["type"])]
                relations[jdx]["t"] = tmp_dickies[(e2["name"], e2["type"])]

            for ner in NER:
                if (ner["name"], ner["type"]) not in nyNER.keys():
                    nyNER[(ner["name"], ner["type"])] = []
                    nyNER[(ner["name"], ner["type"])].append(ner)

            # Remove duplicates from the dictionary
            nyNER = remove_duplicates_from_dict(nyNER)

            new_nyNER[i] = list(nyNER.values())
            new_relations[i] = relations

    # Update the original DataFrame
    combined_df["nyVertexSet"] = new_nyNER
    combined_df["labels"] = new_relations
    combined_df["vertexSet"] = combined_df["nyVertexSet"]

    # Drop the column "nyNER"
    combined_df.drop(columns=["nyVertexSet"], inplace=True)
    print("Successfully fixed grouped entities")
    print("Columns in the dataframe: ", combined_df.columns)
    return combined_df


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        list_of_files = zip_ref.namelist()
        progress_bar = tqdm(list_of_files)
        for file in progress_bar:
            progress_bar.set_description(f"Extracting {file}")
            zip_ref.extract(member=file, path=extract_to)
    print(f"Files extracted to {extract_to}")


def save_df_in_multiple_files(
    df,
    output_dir,
    base_file_name="combined_dataset_corrected",
    orient="records",
    chunk_size=130000,
):
    """Saves a DataFrame to multiple JSON files with a progress bar, updating for each chunk of data written.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_dir (str): Directory to save the files.
        base_file_name (str): Base name of the JSON files.
        orient (str): Orientation of the JSON data. ('records', 'split', 'index', etc.)
        chunk_size (int): Number of rows per chunk file.
    """
    output_dir = Path(output_dir)  # Ensure output_dir is a Path object
    output_dir.mkdir(
        parents=True, exist_ok=True
    )  # Create the output directory if it doesn't exist

    total_chunks = (
        len(df) + chunk_size - 1
    ) // chunk_size  # Calculate the total number of chunks

    # Save each chunk as a separate file
    for i in tqdm(range(total_chunks), desc="Saving chunks"):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_row:end_row]

        file_name = f"{base_file_name}_part_{i + 1}.json"
        file_path = output_dir / file_name

        # Convert chunk DataFrame to JSON format
        with yaspin(
            text=f"Converting chunk {i + 1} to JSON object", color="yellow"
        ) as spinner:
            spinner.spinner = Spinners.binary
            json_str = chunk_df.to_json(orient=orient, lines=False)
            if json_str:
                spinner.ok("✅ ")
            else:
                spinner.fail("💥 ")

        # Write the JSON string to a file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json_str)

    print(f"Saved {total_chunks} chunks to {output_dir}")


def save_df_no_lines_json(
    df, output_dir, file_name="combined_dataset_corrected.json", orient="records"
):
    """Saves a DataFrame to a JSON file with a progress bar, updating for each chunk of data written.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_dir (str): Directory to save the file.
        file_name (str): Name of the JSON file.
        orient (str): Orientation of the JSON data. ('records', 'split', 'index', etc.)
    """
    file_path = Path(output_dir, file_name)  # Construct the full file path

    # Convert DataFrame to JSON format
    with yaspin(text="Converting to JSON object", color="yellow") as spinner:
        spinner.spinner = Spinners.binary
        json_str = df.to_json(orient=orient, lines=False)
        if json_str:
            spinner.ok("✅ ")
        else:
            spinner.fail("💥 ")

    chunk_size = 1024  # Define the size of each chunk to write (in characters)

    # Initialize the progress bar
    with tqdm(total=len(json_str), desc="Writing JSON", leave=True) as pbar:
        with open(file_path, "w", encoding="utf-8") as f:
            # Write in chunks
            for i in range(0, len(json_str), chunk_size):
                chunk = json_str[i : i + chunk_size]
                f.write(chunk)
                pbar.update(len(chunk))


def write_json_with_progress(df, file_path):
    """
    Write a DataFrame to a JSON file with a progress bar.
    Includes lines to convert DataFrame to JSON lines format.

    Args:
        df (DataFrame): The DataFrame to be written to the JSON file.
        file_path (str): The path to the JSON file to be written.
    """
    # Ensure the directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert DataFrame to JSON lines format
    json_lines = df.to_dict(orient="records")

    # Write the JSON lines to the file with a progress bar
    with open(file_path, "w", encoding="utf-8") as f:
        for record in tqdm(json_lines, desc=f"Writing to {file_path}"):
            json.dump(record, f)
            f.write("\n")


def download_files(base_url, download_dir, files):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    for file in files:
        url = f"{base_url}/{file}"
        response = requests.get(url)
        with open(os.path.join(download_dir, file), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {file}")


def download_file(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    else:
        raise Exception(
            f"Failed to download file with status code: {response.status_code}"
        )


def download_zip(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    print(f"Downloaded file saved to {save_path}")


def download_json_files_from_github(repo_url, download_folder, verbose=False):
    # Create the download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # GitHub repo URL to scrape
    base_url = "https://github.com"
    response = requests.get(repo_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links to JSON files
    json_files = []
    while not json_files:
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".json") and "blob" in href:
                json_files.append(base_url + href.replace("blob", "raw"))
        if json_files:  # If no JSON files found, print a warning
            print(
                Fore.YELLOW + "JSON files found in the GitHub repository -- Downloading"
            )
        if not json_files:
            print(
                Fore.YELLOW
                + "No JSON files found in the GitHub repository -- Retrying in 3 seconds"
            )
            time.sleep(3)

    # Download each JSON file
    for json_url in json_files:
        file_name = json_url.split("/")[-1]
        file_path = os.path.join(download_folder, file_name)
        while not os.path.exists(file_path):  # Keep trying until the file is downloaded
            try:
                download_file(json_url, file_path)
                if verbose:
                    print(Fore.RED + f"Downloaded: {file_name}")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")
    print(Fore.WHITE)  # Set color back to white


# def download_json_files_from_github(repo_url, download_folder, verbose=False):

#     # Create the download folder if it doesn't exist
#     if not os.path.exists(download_folder):
#         os.makedirs(download_folder)

#     # GitHub repo URL to scrape
#     base_url = "https://github.com"
#     response = requests.get(repo_url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     # Find all links to JSON files
#     json_files = []
#     for link in soup.find_all("a", href=True):
#         href = link["href"]
#         if href.endswith(".json") and "blob" in href:
#             json_files.append(base_url + href.replace("blob", "raw"))

#     # Download each JSON file
#     for json_url in json_files:
#         file_name = json_url.split("/")[-1]
#         file_path = os.path.join(download_folder, file_name)
#         if not os.path.exists(file_path):  # Check if file already exists
#             try:
#                 download_file(json_url, file_path)
#                 if verbose:
#                     print(Fore.RED + f"Downloaded: {file_name}")
#             except Exception as e:
#                 print(f"Failed to download {file_name}: {e}")
#     print(Fore.WHITE)  # Set color back to white


def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(response.content)


def download_files_Conll04(base_url, download_dir, files):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    for file in files:
        url = f"{base_url}/{file}"
        response = requests.get(url)
        with open(os.path.join(download_dir, file), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {file}")


def read_json_with_progress_no_lines(file_path):

    # Read the JSON file content with progress bar
    with open(file_path, "r") as file:
        # Determine the file size for the progress bar
        file.seek(0, 2)  # Move to the end of the file
        file_size = file.tell()

        file.seek(0)  # Move back to the start of the file

        # Initialize the progress bar
        pbar = tqdm(total=file_size, desc="Reading JSON", leave=False)

        # Read and parse the JSON
        content = []
        while True:
            chunk = file.read(1024)  # Read in 1KB chunks
            if not chunk:
                break
            content.append(chunk)
            pbar.update(len(chunk))

        pbar.close()
        with yaspin(text="Converting to JSON object", color="yellow") as spinner:
            spinner.spinner = Spinners.binary
            data = json.loads("".join(content))
            if data:
                spinner.ok("✅ ")
            else:
                spinner.fail("💥 ")
        # Parse the accumulated content
        return data


def download_files(url, download_path):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(download_path, "wb") as file:
        file.write(response.content)
    print(f"Downloaded file to {download_path}")


def list_files_in_folder(folder):
    json_files = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.endswith(".json")
    ]
    return json_files


def mapping_types(entity_type):
    types = {
        "LOC": "<loc>",
        "MISC": "<misc>",
        "PER": "<per>",
        "NUM": "<num>",
        "TIME": "<time>",
        "ORG": "<org>",
    }
    return types.get(entity_type, f"<{entity_type}>")


def load_json(file_path):
    """Load JSON data from the given file path.
    Assuming the JSON data is one singular json object

    ex:

    {object}
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def load_json_lines(file_path):
    """Load JSON data from the given file path where each line is a separate JSON object.

    Args:
        file_path (str): The path to the file containing JSON lines.

    Returns:
        list: A list of dictionaries, each representing a JSON object from a line in the file.
    """
    data = []
    with open(file_path, "r") as f:
        # Determine the total number of lines for progress indication
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer to the beginning

        # Read and parse each line with a progress bar
        with tqdm(total=total_lines, desc="Loading JSON Lines", leave=False) as pbar:
            for line in f:
                data.append(json.loads(line))
                pbar.update(1)
    return data


def concatenate_sentences(sentences):
    """Concatenate a list of sentences into a single list of words."""
    return " ".join(sentences).split()


def sentences_to_dict(sentences):
    """Convert a list of sentences into a list of dictionaries with sentence ID and sentence."""
    return [{"sent_id": i, "sent": sentence} for i, sentence in enumerate(sentences)]


def combine_list_of_lists(relations):
    """Combine a list of lists into a single list."""
    combined_list = []
    for sublist in relations:
        combined_list.extend(sublist)
    return combined_list


# def generate_examples_test(filepath, lines=True):
#     """Generate examples in raw text form for testing."""
#     print(f"Generating examples from = {filepath}")
#     examples = []
#     if lines:
#         with open(filepath) as json_file:
#             f = [json.loads(line) for line in json_file]
#     else:
#         with open(filepath) as json_file:
#             f = json.load(json_file)

#     for row in f:
#         triplets = ""
#         prev_head = None
#         relations_sorted = sorted(row["relations"], key=lambda tup: tup["h"])
#         for relation in relations_sorted:
#             if prev_head == relation["h"]:
#                 triplets += (
#                     f' {mapping_types(row["NER"][relation["h"]]["type"])} '
#                     + row["NER"][relation["t"]]["name"]
#                     + f' {mapping_types(row["NER"][relation["t"]]["type"])} '
#                     + relation["r"]
#                 )
#             elif prev_head is None:
#                 triplets += (
#                     "<triplet> "
#                     + row["NER"][relation["h"]]["name"]
#                     + f' {mapping_types(row["NER"][relation["h"]]["type"])} '
#                     + row["NER"][relation["t"]]["name"]
#                     + f' {mapping_types(row["NER"][relation["t"]]["type"])} '
#                     + relation["r"]
#                 )
#                 prev_head = relation["h"]
#             else:
#                 triplets += (
#                     "<triplet> "
#                     + row["NER"][relation["h"]]["name"]
#                     + f' {mapping_types(row["NER"][relation["h"]]["type"])} '
#                     + row["NER"][relation["t"]]["name"]
#                     + f' {mapping_types(row["NER"][relation["t"]]["type"])} '
#                     + relation["r"]
#                 )
#                 prev_head = relation["h"]

#         sentences_combined = " ".join([sent["sent"] for sent in row["sentences"]])
#         examples.append(
#             {
#                 "title": row["title"],
#                 "sentences": sentences_combined,
#                 "id": row["title"],
#                 "triplets": triplets,
#             }
#         )
#     return examples


# def save_df(df, modified_path):
#     df.to_json(modified_path, orient="records", lines=True)


def save_json_with_progress(df, file_path, progress_bar=True):
    # Define the total number of rows in the DataFrame for the progress bar
    total_rows = len(df)

    # Open the file path for writing. Ensure it's in write mode 'w'
    with open(file_path, "w", encoding="utf-8") as file:
        if progress_bar:
            # Initialize tqdm progress bar
            pbar = tqdm(total=total_rows, desc=f"Writing to {file_path}")

        # Iterate over DataFrame rows
        for index, row in df.iterrows():
            # Convert row to JSON format and write to file
            file.write(
                row.to_json() + "\n"
            )  # Adding '\n' to create a new line for each JSON object

            # Update the progress bar if it's enabled
            if progress_bar:
                pbar.update(1)

        # Close the progress bar upon completion
        if progress_bar:
            pbar.close()
    return df


def dict_NER(ners, elongated_sentences):
    """Transform NER data into a list of dictionaries with entity details."""
    lists = []
    for i, ner_for_sentence in enumerate(ners):
        for ner in ner_for_sentence:
            words = " ".join(elongated_sentences[ner[0] : ner[1] + 1])
            lists.append(
                {"name": words, "pos": [ner[0], ner[1]], "sent_id": i, "type": ner[2]}
            )
    return lists
