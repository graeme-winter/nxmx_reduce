# nxmx_reduce

Shrink an NXmx data set to its first *N* images.

Modern macromolecular-crystallography detectors (DECTRIS Eiger2 and friends)
write NXmx/HDF5 data sets that are tens of gigabytes per run. `nxmx_reduce`
makes a small, **self-consistent** copy that keeps only the first *N* frames —
handy for test fixtures, CI, bug reports, tutorials, or emailing a data set to
a colleague without shipping the whole thing.

The copy is a complete NXmx data set: the master file plus trimmed copies of
every file it links to. It opens and processes in DIALS/dxtbx (and other NXmx
readers) exactly like the original, just shorter.

## What it does

Given a master file and `-n N`, it writes into an output directory:

1. **Image stack truncated to the first `N` frames.** Compressed image chunks
   are copied **byte-for-byte** (`read_direct_chunk`/`write_direct_chunk`) —
   they are never decompressed and re-compressed, so it is fast and the pixels
   are identical to the original. Data files that fall entirely past the cut
   are not copied at all, and virtual data sets (VDS) are rebuilt to point at
   the trimmed files.
2. **Masks recompressed.** `pixel_mask` and similar are rewritten with
   gzip+shuffle, typically shrinking a multi-megabyte mask to a few tens of kB.
3. **Per-image arrays trimmed.** Goniometer angles, per-frame timestamps,
   counts, etc. — anything whose length matches the original frame count — is
   cut to `N`, and book-keeping scalars like `nimages`/`image_nr_high` are
   patched to match.

## Requirements

- Python 3.9+
- [`h5py`](https://www.h5py.org/) and `numpy`
- [`hdf5plugin`](https://pypi.org/project/hdf5plugin/) — **optional**. Only
  needed for `--verify` (which reads frames back and therefore has to
  decompress them). The reduction itself copies compressed chunks without
  decoding, so it does not need the plugin.

```
pip install h5py numpy hdf5plugin
```

No build step; it is a single file.

## Usage

```
python nxmx_reduce.py -n N MASTER [-o OUTDIR] [--verify] [-v]
```

Keep the first 600 images of `ins10_1.nxs`, writing to `./small`, and read the
frames back to check them:

```
python nxmx_reduce.py -n 600 ins10_1.nxs -o small --verify -v
```

The master may be named `*_master.h5`, `*.nxs`, or anything else — pass
whichever file the beamline gave you as the top of the data set.

Then point your software at the copy exactly as you would the original:

```
dials.import small/ins10_1.nxs
```

### Options

| Option | Meaning |
|---|---|
| `-n, --num-images N` | **(required)** number of images to keep, from the start of the scan |
| `-o, --output DIR` | output directory (default: `./reduced`) |
| `--verify` | after writing, read every retained frame back and compare it with the original (needs a compression filter plugin, i.e. `hdf5plugin`) |
| `-l, --compression-level 0–9` | gzip level for recompressed masks (default: 4) |
| `--also-compress GLOB` | also gzip an extra data set by name, e.g. `--also-compress flatfield` (repeatable) |
| `--libver v108…v114\|latest` | upper HDF5 format bound for the output (default: `v110`, the oldest that supports VDS — see the caveat below) |
| `--no-fixup` | do not patch `nimages`/`image_nr_high` book-keeping scalars |
| `-v, --verbose` | narrate what is being copied, trimmed, and recompressed |

## Example output

```
images    : 3600 -> 600
files     : 6 in data set, 3 written
masks     : 1 recompressed
per-image : 16 data set(s) truncated
size      : 10.2 GB -> 1.8 GB  (5.8x smaller)
output    : /path/to/small/ins10_1.nxs

verifying ...
  ok   ins10_1_000001.h5:/data: 600 frames
  ok   /entry/data/data: 600 frames identical
```

How much smaller depends on the data: for sparse frames the drop is dramatic;
for dense, already well-compressed diffraction data the per-frame size is
essentially fixed, so the reduction is roughly proportional to `N / total`.

## Notes and caveats

- **Always try `--verify` on a new kind of data set.** It reads every kept
  frame back through the (possibly virtual) image stack and compares it with
  the original. It is cheap insurance against a subtly wrong copy.
- **`--verify` needs `hdf5plugin`** (or another plugin providing the data
  set's compression filter), because reading frames back means decompressing
  them. The reduction step itself does not.
- **Frames are kept from the start of the scan** (`1 … N`). Selecting an
  arbitrary sub-range is not supported.
- **Do not pass `--libver latest` unless every program that will read the copy
  is as new as the one writing it.** The default (`v110`) produces output that
  older HDF5 libraries (e.g. the one shipped with current DIALS) can read.
- **Different filter-plugin version?** If the copy is made with a bitshuffle
  (or other) plugin whose version differs from the one that wrote the data,
  the exact filter recipe cannot always be recreated. The chunks are still
  copied verbatim; the filter is just stored as *optional* rather than
  *mandatory*. This is invisible to any reader that has the plugin installed
  (all normal readers do).

## Testing

```
python make_eiger.py eiger                                  # synthetic data set
python nxmx_reduce.py -n 12 eiger/lyso_1_master.h5 -o out --verify -v
python test_filters.py                                      # every installable HDF5 filter (needs hdf5plugin)
./test_dials.sh <real_master.h5> [nimages]                  # end-to-end DIALS check on real data
```

`test_dials.sh` reduces a real data set, then confirms that `dials.import`
geometry and `dials.find_spots` counts match `dials.import image_range=1,N` on
the original — i.e. the reduced copy is indistinguishable from the original
over the frames it keeps.
