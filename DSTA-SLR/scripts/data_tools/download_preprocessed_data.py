import argparse
import asyncio
import time
import shutil
import tempfile
import types
from pathlib import Path

import requests
from script_utils import find_repo_root

asyncio.coroutine = types.coroutine

from mega.mega import (  # noqa: E402
    AES,
    Counter,
    a32_to_str,
    base64_to_a32,
    base64_url_decode,
    decrypt_attr,
    decrypt_key,
    get_chunks,
    str_to_a32,
)

ROOT = find_repo_root(__file__)
DATA_ROOT = ROOT / "data"
FOLDER_HANDLE = "EvkEzIAC"
FOLDER_KEY = "gq_nWLbbWoj9WVnJGxnGaA"
API_URL = f"https://g.api.mega.co.nz/cs?id=0&n={FOLDER_HANDLE}"
SESSION = requests.Session()
SESSION.trust_env = False


def fetch_folder_listing():
    response = SESSION.post(
        API_URL,
        json=[{"a": "f", "c": 1, "ca": 1, "r": 1}],
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()[0]
    root_handle = payload["f"][0]["h"]
    root_key = base64_to_a32(FOLDER_KEY)
    processed = {}
    for node in payload["f"]:
        key_parts = dict(
            part.split(":", 1) for part in node.get("k", "").split("/") if ":" in part
        )
        encrypted_key = key_parts.get(root_handle)
        if not encrypted_key:
            continue
        node_key = decrypt_key(base64_to_a32(encrypted_key), root_key)
        if node["t"] == 0:
            node["iv"] = node_key[4:6] + (0, 0)
            node["meta_mac"] = node_key[6:8]
            node["k"] = (
                node_key[0] ^ node_key[4],
                node_key[1] ^ node_key[5],
                node_key[2] ^ node_key[6],
                node_key[3] ^ node_key[7],
            )
        else:
            node["k"] = node_key
        node["a"] = decrypt_attr(base64_url_decode(node["a"]), node["k"])
        processed[node["h"]] = node
    return root_handle, processed


def fetch_file_info(node_handle):
    response = SESSION.post(
        API_URL,
        json=[{"a": "g", "g": 1, "n": node_handle}],
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()[0]
    if "g" not in payload:
        raise RuntimeError(f"Cannot fetch download link for node {node_handle}: {payload}")
    return payload


def download_file(node, destination: Path):
    file_info = fetch_file_info(node["h"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    response = SESSION.get(file_info["g"], stream=True, timeout=120)
    if response.status_code == 509:
        raise RuntimeError(
            f"MEGA bandwidth quota exceeded while downloading {destination.name}"
        )
    response.raise_for_status()
    input_file = response.raw
    k = node["k"]
    iv = node["iv"]
    meta_mac = node["meta_mac"]

    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as temp_output_file:
        temp_name = temp_output_file.name
        k_str = a32_to_str(k)
        counter = Counter.new(128, initial_value=((iv[0] << 32) + iv[1]) << 64)
        aes = AES.new(k_str, AES.MODE_CTR, counter=counter)

        mac_str = "\0" * 16
        mac_encryptor = AES.new(k_str, AES.MODE_CBC, mac_str.encode("utf8"))
        iv_str = a32_to_str([iv[0], iv[1], iv[0], iv[1]])

        for _, chunk_size in get_chunks(file_info["s"]):
            chunk = input_file.read(chunk_size)
            chunk = aes.decrypt(chunk)
            temp_output_file.write(chunk)

            encryptor = AES.new(k_str, AES.MODE_CBC, iv_str)
            last_index = 0
            for last_index in range(0, len(chunk) - 16, 16):
                block = chunk[last_index : last_index + 16]
                encryptor.encrypt(block)
            if file_info["s"] > 16:
                last_index += 16
            block = chunk[last_index : last_index + 16]
            if len(block) % 16:
                block += b"\0" * (16 - (len(block) % 16))
            mac_str = mac_encryptor.encrypt(encryptor.encrypt(block))

        file_mac = str_to_a32(mac_str)
        if len(file_mac) >= 4:
            if (file_mac[0] ^ file_mac[1], file_mac[2] ^ file_mac[3]) != meta_mac:
                raise ValueError(f"Mismatched mac for {destination.name}")

    shutil.move(temp_name, destination)
    return destination


def normalize_filename(filename):
    return (
        filename.replace("test_data_joint.npy", "val_data_joint.npy")
        .replace("test_label.pkl", "val_label.pkl")
        .replace("test_labels.csv", "val_labels.csv")
    )


def normalize_foldername(foldername):
    return foldername.replace("NMFs_CSL_skeleton", "NMFs-CSL")


def build_children_map(nodes):
    children = {}
    for node in nodes.values():
        parent = node.get("p")
        if parent is None:
            continue
        children.setdefault(parent, []).append(node)
    return children


def find_folder_by_name(nodes, name):
    for node in nodes.values():
        if node["t"] == 1 and node["a"].get("n") == name:
            return node
    raise KeyError(f"Folder {name} not found")


def download_tree(node, children_map, target_root: Path):
    if node["t"] == 1:
        folder_path = target_root / normalize_foldername(node["a"]["n"])
        folder_path.mkdir(parents=True, exist_ok=True)
        for child in children_map.get(node["h"], []):
            download_tree(child, children_map, folder_path)
    else:
        destination = target_root / normalize_filename(node["a"]["n"])
        if destination.exists() and destination.stat().st_size > 0:
            return
        print(f"Downloading {destination}")
        last_error = None
        for _ in range(3):
            try:
                download_file(node, destination)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(2)
        if last_error is not None:
            raise last_error


def main():
    parser = argparse.ArgumentParser(
        description="Download selected preprocessed dataset folders from the public DSTA-SLR MEGA folder."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset folder names in the public MEGA folder, e.g. MSASL100 SLR500 NMFs_CSL_skeleton",
    )
    args = parser.parse_args()

    _, nodes = fetch_folder_listing()
    children_map = build_children_map(nodes)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        folder = find_folder_by_name(nodes, dataset_name)
        download_tree(folder, children_map, DATA_ROOT)


if __name__ == "__main__":
    main()
