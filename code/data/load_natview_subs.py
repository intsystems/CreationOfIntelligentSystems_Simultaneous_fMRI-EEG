import os
import concurrent.futures
import requests
import glob

def download_file(url):
    cmd = f"wget -q {url} -P ./natview/data/"
    os.system(cmd)

def parallel_download(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(download_file, url): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading {url}: {e}")

os.makedirs('./natview/data', exist_ok=True)
url_template = 'https://fcp-indi.s3.amazonaws.com/data/Projects/NATVIEW_EEGFMRI/preproc_data_gz/sub-{:02d}.tar.gz'
urls = [url_template.format(idx) for idx in range(1, 23)]
parallel_download(urls)