import argparse
import csv
import dataclasses
import datetime
import itertools
import os
import pathlib
import random
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Collection, Iterable
from http.client import HTTPResponse
from io import BufferedReader
from typing import Optional
from urllib import request

try:
    import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

DOWNLOAD_URL: str = "https://www.openslr.org/resources/12/test-clean.tar.gz"
CORPUS_DIR: pathlib.Path = pathlib.Path(__file__).parent / "tests/corpus"

if HAS_TQDM:

    class ProgressBar(tqdm.tqdm):
        pass

else:

    class ProgressBar:
        def __init__(
            self,
            iterable: Optional[Iterable | Collection] = None,
            total=None,
            desc=None,
            unit: str = "it",
            *args,
            **kwargs,
        ):
            self.iter = iter(iterable) if iterable is not None else None

            if total:
                self.total = total
            elif isinstance(iterable, Collection):
                self.total = len(iterable)
            else:
                self.total = None

            self.desc = desc
            self.unit = unit
            self.n = 0

        def __iter__(self):
            return self

        def __next__(self):
            try:
                n = next(self.iter)
                self._update()
            except StopIteration:
                self._flush()
                raise
            return n

        def update(self, n=1):
            self.n += n
            self._render()

        def close(self):
            pass

        def _flush(self):
            print("", flush=True)

        def _render(self):
            if self.total:
                of_total = f" / {int(self.total)}" if self.total else ""
            desc = self.desc + ": " if self.desc else ""
            progress = int(self.n)
            print(
                f"{desc}{progress}{of_total} {self.unit}",
                flush=True,
                end="\r",
            )


@dataclasses.dataclass
class Fragment:
    path: pathlib.PurePath
    duration: datetime.timedelta
    transcription: str


class DownloadProgress(BufferedReader):
    def __init__(self, response, *args, **kwargs):
        self.bytes_read = 0
        self.progress_bar = ProgressBar(
            total=response.length, unit="Byte", unit_scale=True, *args, **kwargs
        )
        super().__init__(response)

    def read(self, *args, **kwargs):
        buf = super().read(*args, **kwargs)
        self.progress_bar.update(len(buf))
        return buf


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
        with tarfile.open(fileobj=DownloadProgress(resp), mode="r|*") as tf:
            tf.extractall(path=tmpdir)
    except Exception:
        os.rmdir(tmpdir)
        raise
    return tmpdir


def index_dataset(dir: pathlib.Path) -> list[Fragment]:
    chapters = (dir / "LibriSpeech" / "test-clean").glob("*/*/")
    fragments = []
    for chapter_path in ProgressBar(
        list(chapters), desc="Indexing dataset", unit="chapter"
    ):
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


def create_testset(
    fragments: list[Fragment], total_duration_hours: int, output_dir: pathlib.Path
):
    duration_limit = datetime.timedelta(hours=total_duration_hours)
    total_duration = datetime.timedelta()
    progress = ProgressBar(
        total=duration_limit.total_seconds(), desc="Creating test set", unit="second"
    )
    for i in itertools.count():
        if total_duration >= duration_limit:
            progress.close()
            return

        audio_file = output_dir / f"{i}.flac"
        transcription_file = output_dir / f"{i}.csv"

        k = random.randint(10, min(len(fragments), 300))
        parts: list[Fragment] = random.sample(fragments, k=k)
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
        progress.update(duration.total_seconds())


def dir_path(string):
    path = pathlib.Path(string)
    if path.is_dir():
        return path
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "-d",
        "--duration",
        type=float,
        help="Total duration of test samples in hours",
        default=100,
    )
    argp.add_argument("-o", "--output-dir", type=dir_path, default=str(CORPUS_DIR))

    args = argp.parse_args(sys.argv[1:])
    corpus_dir = args.output_dir
    duration = args.duration
    dataset_dir = download_dataset(DOWNLOAD_URL)
    fragments = index_dataset(dataset_dir)
    create_testset(fragments, total_duration_hours=duration, output_dir=corpus_dir)
