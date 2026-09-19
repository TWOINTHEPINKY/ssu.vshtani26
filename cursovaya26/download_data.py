from huggingface_hub import hf_hub_download
import os
import re

REPO_ID = "deepvk/VK-LSVD"
SUBSAMPLE = "up0.001_ip0.001"
LAST_WEEK_TO_DOWNLOAD = 2

print(f"Загрузка недель с 00 по {LAST_WEEK_TO_DOWNLOAD} из подвыборки '{SUBSAMPLE}'...")

for week_num in range(LAST_WEEK_TO_DOWNLOAD + 1):
    week_str = f"{week_num:02d}"
    file_path = f"subsamples/{SUBSAMPLE}/train/week_{week_str}.parquet"
    print(f"Загружаю {file_path}...")
    try:
        downloaded_file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir="./vk_data"
        )
        print(f"  -> OK: {downloaded_file_path}\n")
    except Exception as e:
        print(f"  -> Ошибка при загрузке {file_path}: {e}\n")