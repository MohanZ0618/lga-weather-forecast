"""Fingerprint the original S3 objects used by the fixed-time replay, with resumable checks."""
from base64 import b64decode
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import re
import sqlite3
from threading import local
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data import OUT, STORE, SNAPSHOT_END

THREAD = local()


def fingerprint(url):
    if not hasattr(THREAD,"client"):
        THREAD.client = requests.Session()
        retry = Retry(total=3,backoff_factor=.4,status_forcelist=[429,500,502,503,504])
        THREAD.client.mount("https://",HTTPAdapter(max_retries=retry))
    client = THREAD.client
    response = client.head(url,timeout=(8,30))
    response.raise_for_status()
    headers = response.headers
    tag = headers.get("ETag","").strip('"')
    # For a single-part S3 object using plain storage or SSE-S3 the ETag is its MD5.
    if re.fullmatch(r"[0-9a-fA-F]{32}",tag) and headers.get("x-amz-server-side-encryption") in (None,"AES256"):
        algorithm,value = "md5",tag.lower()
    elif headers.get("x-amz-checksum-type")=="FULL_OBJECT" and "x-amz-checksum-sha256" in headers:
        algorithm,value = "sha256",b64decode(headers["x-amz-checksum-sha256"]).hex()
    else:
        with client.get(url,headers={"If-Match":headers["ETag"]},stream=True,timeout=(8,60)) as body:
            body.raise_for_status()
            checksum = sha256()
            for chunk in body.iter_content(1024*1024):
                checksum.update(chunk)
        algorithm,value = "sha256",checksum.hexdigest()
    return (url,algorithm,value,int(headers["Content-Length"]),headers["Last-Modified"])


def source_checks(watch=False):
    database = sqlite3.connect(STORE/"source_checks.sqlite")
    database.execute("create table if not exists objects (url text primary key, algorithm text, digest text, bytes integer, modified text)")
    known = {r[0] for r in database.execute("select url from objects")}
    visited = set()
    with ThreadPoolExecutor(max_workers=28) as pool:
        while True:
            files = sorted((STORE/"days").glob("*.json"))
            urls = set()
            for path in files:
                if path in visited:
                    continue
                records = json.loads(path.read_text())
                urls.update(r[col] for r in records for col in ("cycle_url","cloud_url","goes_url") if col in r)
                if all(not r["errors"] for r in records):
                    visited.add(path)
            pending = sorted(urls-known)
            for i in range(0,len(pending),200):
                batch = list(pool.map(fingerprint,pending[i:i+200]))
                with database:
                    database.executemany("insert or ignore into objects values (?,?,?,?,?)",batch)
                known.update(r[0] for r in batch)
            if not watch or len(visited)>=len(pd.date_range("2021-01-01",SNAPSHOT_END)):
                break
            time.sleep(15)
    df = pd.read_sql_query("select * from objects order by url",database)
    database.close()
    df.to_csv(OUT/"upstream_hashes.csv.gz",index=False)
    print(len(df),"fingerprinted upstream objects")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch",action="store_true")
    source_checks(parser.parse_args().watch)
