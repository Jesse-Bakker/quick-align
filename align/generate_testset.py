from typing import List
from urllib import request
from http.client import HTTPResponse
import tempfile
import tarfile
import subprocess
import pathlib
import dataclasses
import datetime
import os
from contextlib import redirect_stdout
import sys
import random
import itertools
import csv
import tqdm

DOWNLOAD_URL: str = "https://www.openslr.org/resources/12/test-clean.tar.gz"
CORPUS_DIR: pathlib.Path = pathlib.Path(__file__).parent / "tests/corpus"


@dataclasses.dataclass
class Fragment:
    path: pathlib.PurePath
    duration: datetime.timedelta
    transcription: str


def download_dataset(url: str) -> pathlib.Path:
    prefix = "shoutsync-dl-"
    req = request.Request(url=DOWNLOAD_URL)
    tmp = pathlib.Path(tempfile.gettempdir())
    if existing := list(tmp.glob(prefix + "*")):
        return pathlib.Path(existing[0])
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    try:
        resp: HTTPResponse = request.urlopen(req)
        # Stream through tar with transparent compression
        with tarfile.open(fileobj=resp, mode="r|*") as tf:
            tf.extractall(path=tmpdir)
    except:
        os.rmdir(tmpdir)
        raise
    return tmpdir


def index_dataset(dir: pathlib.Path) -> List[Fragment]:
    chapters = (dir / "LibriSpeech" / "test-clean").glob("*/*/")
    fragments = []
    for chapter_path in tqdm.tqdm(list(chapters), desc="Indexing dataset"):
        chapter_path = pathlib.Path(chapter_path)
        book, chap = chapter_path.parts[-2:]
        trans = chapter_path / f"{book}-{chap}.trans.txt"
        with trans.open() as trans_f:
            for line in trans_f:
                name, text = line.split(" ", maxsplit=1)
                path = chapter_path / (name + ".flac")
                ffprobe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        path,
                    ],
                    capture_output=True,
                    encoding="ascii",
                    check=True,
                )
                duration = float(ffprobe.stdout.splitlines()[0])
                fragments.append(
                    Fragment(
                        path=path,
                        transcription=text,
                        duration=datetime.timedelta(seconds=duration),
                    )
                )
    return fragments


def create_testset(fragments: List[Fragment], total_duration_hours: int):
    duration_limit = datetime.timedelta(hours=total_duration_hours)
    total_duration = datetime.timedelta()
    CORPUS_DIR.mkdir(exist_ok=True)
    progress = tqdm.tqdm(total=duration_limit.total_seconds(), desc="Creating test set")
    for i in itertools.count():
        if total_duration >= duration_limit:
            return

        audio_file = CORPUS_DIR / f"{i}.flac"
        transcription_file = CORPUS_DIR / f"{i}.csv"

        k = random.randint(10, min(len(fragments), 300))
        parts: List[Fragment] = random.sample(fragments, k=k)
        files = "|".join(str(part.path) for part in parts)
        subprocess.run(
            ["ffmpeg", "-i", "concat:" + files, "-c", "copy", str(audio_file)],
            check=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        with open(transcription_file, "w") as f:
            writer = csv.writer(f)
            duration = datetime.timedelta()
            for part in parts:
                writer.writerow([duration.total_seconds(), part.transcription.strip()])
                duration += part.duration
            total_duration += duration
        progress.update(total_duration.total_seconds())


if __name__ == "__main__":
    if CORPUS_DIR.exists():
        with redirect_stdout(sys.stderr):
            print(f"Corpus directory ({CORPUS_DIR}) already exists. Aborting.")
            exit(1)
    dataset_dir = download_dataset(DOWNLOAD_URL)
    fragments = index_dataset(dataset_dir)
    create_testset(fragments, total_duration_hours=100)
