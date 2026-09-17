"""Bringing a trace onto the machine.

  python -m loop.workloads fetch <url> <out_path> [prefix_mb]
      Download a trace, or only its first prefix_mb megabytes (an HTTP range): a
      100 MB prefix holds ~88M instructions, enough to simulate.
  python -m loop.workloads fetch_gap <zip_url> <member> <out_dir>
      One GAP trace out of a 10 GB Zenodo zip without downloading the zip.
"""

import os
import struct
import sys
import urllib.request
import zlib


def fetch(url, out_path, prefix_mb=None):
    """Download a file, or only its first prefix_mb megabytes with an HTTP range."""
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    if prefix_mb is not None:
        request.add_header("Range", "bytes=0-{}".format(prefix_mb * 1024 * 1024 - 1))
    with urllib.request.urlopen(request) as response, open(out_path, "wb") as out_file:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if len(chunk) == 0:
                break
            out_file.write(chunk)
    print("fetched {} -> {} ({} MB)".format(url, out_path, os.path.getsize(out_path) // (1024 * 1024)), flush=True)


def fetch_gap(zip_url, member_name, out_dir):
    """Read the zip64 central directory from the archive's tail, find the member,
    then byte-range just that member and inflate it."""
    def fetch_range(start, end):
        request = urllib.request.Request(zip_url, headers={"Range": "bytes={}-{}".format(start, end),
                                                            "User-Agent": "curl/8"})
        with urllib.request.urlopen(request) as response:
            return response.read()

    with urllib.request.urlopen(urllib.request.Request(zip_url, method="HEAD")) as response:
        total = int(response.headers["Content-Length"])
    tail = fetch_range(total - 256 * 1024, total - 1)
    tail_start = total - len(tail)
    # zip64 end-of-central-directory locator: where the central directory starts.
    locator = tail.rfind(b"PK\x06\x07")
    if locator >= 0:
        end_record_offset = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        end_record = fetch_range(end_record_offset, end_record_offset + 56)
        directory_size = struct.unpack("<Q", end_record[40:48])[0]
        directory_offset = struct.unpack("<Q", end_record[48:56])[0]
    else:
        end = tail.rfind(b"PK\x05\x06")
        directory_size = struct.unpack("<I", tail[end + 12:end + 16])[0]
        directory_offset = struct.unpack("<I", tail[end + 16:end + 20])[0]
    if directory_offset >= tail_start:
        directory = tail[directory_offset - tail_start:directory_offset - tail_start + directory_size]
    else:
        directory = fetch_range(directory_offset, directory_offset + directory_size - 1)
    position = 0
    while position < len(directory):
        if directory[position:position + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", directory[position + 10:position + 12])[0]
        compressed_size = struct.unpack("<I", directory[position + 20:position + 24])[0]
        name_length = struct.unpack("<H", directory[position + 28:position + 30])[0]
        extra_length = struct.unpack("<H", directory[position + 30:position + 32])[0]
        comment_length = struct.unpack("<H", directory[position + 32:position + 34])[0]
        local_offset = struct.unpack("<I", directory[position + 42:position + 46])[0]
        name = directory[position + 46:position + 46 + name_length].decode()
        extra = directory[position + 46 + name_length:position + 46 + name_length + extra_length]
        # zip64 extra field: 64-bit sizes and offset where the 32-bit fields are saturated.
        extra_position = 0
        while extra_position + 4 <= len(extra):
            field_id, field_size = struct.unpack("<HH", extra[extra_position:extra_position + 4])
            if field_id == 1:
                field = extra[extra_position + 4:extra_position + 4 + field_size]
                values = []
                for index in range(0, len(field) - 7, 8):
                    values.append(struct.unpack("<Q", field[index:index + 8])[0])
                cursor = 0
                if struct.unpack("<I", directory[position + 24:position + 28])[0] == 0xFFFFFFFF:
                    cursor += 1
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = values[cursor]
                    cursor += 1
                if local_offset == 0xFFFFFFFF:
                    local_offset = values[cursor]
            extra_position += 4 + field_size
        position += 46 + name_length + extra_length + comment_length
        if os.path.basename(name) != member_name:
            continue
        header = fetch_range(local_offset, local_offset + 29)
        header_name_length = struct.unpack("<H", header[26:28])[0]
        header_extra_length = struct.unpack("<H", header[28:30])[0]
        data_start = local_offset + 30 + header_name_length + header_extra_length
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, member_name)
        print("fetching {} ({} MB) ...".format(member_name, compressed_size // (1024 * 1024)), flush=True)
        data = fetch_range(data_start, data_start + compressed_size - 1)
        if method == 8:
            data = zlib.decompress(data, -15)
        with open(out_path, "wb") as out_file:
            out_file.write(data)
        print("saved", out_path, flush=True)
        return out_path
    raise RuntimeError("member not found: " + member_name)


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
    elif sys.argv[1] == "fetch_gap":
        fetch_gap(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
