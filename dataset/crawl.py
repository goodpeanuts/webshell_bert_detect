import os
import subprocess
import shutil
from datetime import datetime
import logging


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("./logs", exist_ok=True)


def git_clone(REPOS: list[tuple[str, str, str]], data_type: str):
    skip = 0
    error: list[tuple[str, str, str]] = []
    success: list[tuple[str, str, str, str]] = []

    BASE_DIR = os.path.join("./repo", data_type)
    os.makedirs(BASE_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", filename=f"./logs/{data_type}-{current_time}.log", filemode="w")


    for user, repo, description in REPOS:
        folder_name = f"{user}-{repo}"
        dest_path = os.path.join(BASE_DIR, folder_name)

        if os.path.exists(dest_path):
            logging.info(f"[SKIP] {folder_name} already exists")
            skip += 1
            continue

        url = f"https://github.com/{user}/{repo}.git"
        logging.info(f"[Clone] {url} -> {dest_path}, decription: {description}")

        try:
            subprocess.run(["git", "clone", url, dest_path], check=True)

            hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=dest_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            success.append((user, repo, description, hash.stdout.strip()))

            git_dir = os.path.join(dest_path, ".git")
            if os.path.isdir(git_dir):
                shutil.rmtree(git_dir)
                logging.info(f"[HASH]: {hash}, remove {folder_name}  .git/ ")
            

        except subprocess.CalledProcessError as e:
            error.append((user, repo, description))
            logging.error(f"failed to clone {url}: ({e})")

    logging.info(f"successfully cloned {len(success)} repos, skipped {skip} repos")
    if len(success) > 0:
        logging.info("successfully cloned repos:")
        for user, repo, description, hash in success:
            logging.info(f"{user}-{repo} [hash]: {hash}, [decription]: {description} ")
    
    if error.__len__() > 0:
        logging.error(f"{error.__len__()} repos failed to clone")
        for user, repo, description in error:
            logging.error(f"[failed] {user}-{repo}, [decription]: {description}")