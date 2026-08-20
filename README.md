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

Optionally, `--frames-per-data N` **repartitions** the kept stack into output
data files of at most `N` frames each (instead of mirroring the source's
partitioning), rebuilding the virtual data set and the `data_00000i` links to
match. See [Repartitioning](#repartitioning-the-data-files) below.

Optionally, `--crop` **cuts out the central area of an Eiger detector**
(16M→4M, 9M→1M by default) and rewrites the geometry header — image size, beam
centre and detector origin — so the smaller detector reads back correctly. See
[Cropping the detector](#cropping-the-detector) below.

## Requirements

- Python 3.9+
- [`h5py`](https://www.h5py.org/) and `numpy`
- [`hdf5plugin`](https://pypi.org/project/hdf5plugin/) — **optional**. Needed
  for `--verify` (which reads frames back and therefore has to decompress them)
  and for `--crop` (which decodes every frame to cut out the sub-window). The
  plain reduction copies compressed chunks without decoding, so it does not
  need the plugin.

## Installation

```
pip install nxmx-reduce
```

This installs the `nxmx-reduce` command and pulls in `h5py` and `numpy`. To
also get `hdf5plugin` (needed only for `--verify`):

```
pip install "nxmx-reduce[plugins]"
```

To install from a checkout:

```
pip install .
```

It is still a single-file module — you can also just run `nxmx_reduce.py`
directly without installing, provided `h5py` and `numpy` are available.

## Usage

```
nxmx-reduce -n N MASTER [-o OUTDIR] [--verify] [-v]
```

(or, without installing, `python nxmx_reduce.py -n N MASTER …`)

Keep the first 600 images of `ins10_1.nxs`, writing to `./small`, and read the
frames back to check them:

```
nxmx-reduce -n 600 ins10_1.nxs -o small --verify -v
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
| `--frames-per-data N` | repartition the kept images into data files of at most `N` frames each, rebuilding the VDS and `data_00000i` links (DECTRIS self-referencing-VDS layout) |
| `--crop` | crop the central detector area to a smaller Eiger class (16M→4M, 9M→1M), adjusting the header; decodes frames, so needs `hdf5plugin` |
| `--crop-to CLASS` | crop to a specific class instead of the default (`500K`, `1M`, `4M`, `9M`, `16M`); implies `--crop` |
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

## Repartitioning the data files

By default the output keeps the source's own image-file partitioning (one
trimmed data file per source data file). With `--frames-per-data N` the kept
stack is instead re-sliced into files of at most `N` frames each:

```
nxmx-reduce -n 600 --frames-per-data 100 ins10_1.nxs -o small --verify
```

This writes six data files of 100 frames — `ins10_1_000001.h5 …
ins10_1_000006.h5`, continuing your data set's own naming — plus a master whose
`entry/data/data` is a self-referencing VDS over regenerated `data_00000i`
links, i.e. byte-structurally the same shape as a native DECTRIS master, just
repartitioned. When the counts line up the filenames are identical to the
originals.

`N` can be larger than the number of kept frames (everything lands in one file)
or as small as `1` (one frame per file). A block may span two source files —
they are stitched seamlessly.

It currently supports the DECTRIS self-referencing-VDS-over-external-link
layout (the usual NE-CAT Eiger2 shape). Other layouts are rejected with a clear
message; omit the flag to reduce them with their partitioning preserved.

## Cropping the detector

`--crop` cuts out the central rectangle of an Eiger detector, keeping a whole
number of the inner modules and discarding the outer ones (and their gaps):

```
nxmx-reduce -n 20 --crop ins10_1.nxs -o small --verify
```

By default the target class is chosen from the source: a **16M → 4M** and a
**9M → 1M**. Force another target with `--crop-to`:

```
nxmx-reduce -n 20 --crop-to 1M ins10_1.nxs -o small --verify
```

An Eiger image is a grid of identical modules separated by fixed gaps. Two
module geometries are supported, and the right one is picked automatically from
the image dimensions (they never collide):

| Family | Module (fast×slow) | Gaps (fast/slow) | 16M size (fast×slow) |
|---|---|---|---|
| Eiger (gen 1) | 1030×514 | 10 / 37 | 4150×4371 |
| Eiger2 | 1028×512 | 12 / 38 | 4148×4362 |

The crop always starts and ends on a module boundary, so it only ever removes
whole outer modules — no module is split. The header is rewritten to match so
the copy is geometrically correct in DIALS and other NXmx readers:

- the image size (`x/y_pixels_in_detector`, `module/data_size`);
- the beam centre (`beam_center_x/y`), shifted by the pixels removed from the
  top-left corner;
- the detector origin (`module/module_offset`), moved along the fast/slow pixel
  directions so the module still sits at the right place in the lab frame. The
  physical beam position is invariant under the crop — DIALS derives the beam
  centre from `module_offset` and gets exactly the written `beam_center_*`.

Detector-sized 2-D maps (masks, flatfields) are cropped to the same window;
per-image 1-D arrays (omega, timestamps, …) are untouched.

Unlike the plain reduction, cropping takes a sub-window of each frame — which
is not a whole compressed chunk — so frames are **decoded, sliced and
recompressed** (gzip+shuffle) rather than copied verbatim. That means a filter
plugin (`hdf5plugin`) must be installed, and it is slower than a plain reduce.
`--crop` cannot be combined with `--frames-per-data`. `--verify` compares each
output frame with the correspondingly-cropped source frame.

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
