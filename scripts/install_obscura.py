"""Install the checksum-pinned Obscura CLI; no Chrome or Python dependencies."""
import hashlib
import io
from pathlib import Path
import platform
import sys
import tarfile
from urllib.request import urlopen
import zipfile

VERSION = 'v0.2.3'
ARCHIVES = {
    ('Windows', 'AMD64'): ('obscura-x86_64-windows-no-render-stealth.zip', 'f3890f8f89c9e7fcdc56941d0a5920e78c5d7b744d5e081bc5a9f3697467e65b'),
    ('Linux', 'x86_64'): ('obscura-x86_64-linux-no-render-stealth.tar.gz', '56239859ef7bc34439013834fd0491ff21b0d0debe372d5a6e373dd025ea3198'),
    ('Linux', 'aarch64'): ('obscura-aarch64-linux-no-render-stealth.tar.gz', 'd0d443c4ebfcf41b29582f4637fe0d3e266a1dbff5eaa877af5c788e8fb1fb8c'),
}


def install(destination):
    name, digest = ARCHIVES[(platform.system(), platform.machine())]
    with urlopen(f'https://github.com/h4ckf0r0day/obscura/releases/download/{VERSION}/{name}', timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Obscura release checksum mismatch')
    destination.mkdir(parents=True, exist_ok=True)
    suffix = '.exe' if name.endswith('.zip') else ''
    wanted = {'obscura' + suffix, 'obscura-worker' + suffix}
    archive = zipfile.ZipFile(io.BytesIO(data)) if suffix else tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')
    with archive:
        members = archive.namelist() if suffix else [m.name for m in archive.getmembers() if m.isfile()]
        for member in members:
            filename = Path(member).name
            if filename in wanted:
                body = archive.read(member) if suffix else archive.extractfile(member).read()
                target = destination / filename
                target.write_bytes(body)
                target.chmod(0o755)
                wanted.remove(filename)
    if wanted:
        raise ValueError('Obscura archive is missing executables')
    print(f'Installed Obscura {VERSION} in {destination.resolve()}')


if __name__ == '__main__':
    install(Path(sys.argv[1]))
