# nxmx_reduce — project context

Tool that shrinks an NXmx data set: recompresses masks, truncates the image
stack to `-n` frames, and truncates every per-image data set to match.
Written for NE-CAT (APS 24-ID-C / 24-ID-E) Eiger2 data.

## Layout

    nxmx_reduce.py    the tool
    pyproject.toml    packaging metadata (installs the `nxmx-reduce` command)

## What it does

Writes a complete, self-consistent copy of the data set into an output
directory: master plus trimmed copies of every file it links to. Image
frames move via `read_direct_chunk` / `write_direct_chunk`, so compressed
chunks are copied byte-for-byte and never decoded — no filter plugin
needed, no recompression cost.

## Non-obvious things that already bit us — do not regress these

1. **Do not clone the source DCPL to create the destination data set.**
   A filter's `set_local` callback runs again at `H5Dcreate` time and
   rewrites `cd_values`. bitshuffle splices in the element size, so the
   values stored in a file are *not* the values you pass to recreate it:

       source   cd_values=(0, 4, 4, 0, 2)            valid bslz4
       naive copy cd_values=(0, 4, 4, 0, 4, 4, 0, 2) undecodable

   The chunks are byte-perfect; the recipe is junk. Symptom on read:
   `Non integer number of elements` or `filter returned failure during
   read`. `PipelineResolver` searches for the pre-`set_local` input by
   creating probe data sets in an in-memory file until one round-trips to
   the exact target pipeline. `copy_frames` asserts the created pipeline
   matches before writing a chunk. Keep that assert.

2. **Never write with `libver="latest"`.** On HDF5 2.0 that emits v5
   chunked-layout messages. HDF5 1.14 (what DIALS ships) opens the master
   fine and then fails on the data files with `bad version number for
   layout message` — so it looks like it worked until DIALS reads images.
   Default is `("earliest", "v110")`, the oldest bound supporting VDS.

3. **VDS source shapes must match the trimmed files exactly.** The
   `VirtualSource` shape is `plan.keep` for that file, not the original
   length; get it wrong and reads silently return fill value instead of
   erroring. `--verify` catches this.

4. **`--verify` is cheap and reads frames back through the VDS.** Run it
   on anything new.

5. **Self-referencing VDS whose source *is* an external link (real DECTRIS
   filewriter layout).** `ins10_1.nxs` maps `/entry/data/data` with source
   file `'.'` and `dset_name=/entry/data/data_000001`, where `data_000001`
   is an *external link* in the master to `/data` in a data file. Two scans
   used to register that file twice — once keyed `(master, data_00000N)` via
   the VDS, once keyed `(realfile, /data)` via the direct external-link scan
   — and `Plan.apply` laid the direct-link copies *after* the VDS copies, so
   every real file got `keep=0`, its link was dropped, and the VDS dangled
   to fill value (`--verify` caught it: "frame 0 differs"). Fix:
   `resolve_extlink` follows the one hop so a self-ref source is keyed
   against its real backing file and the two scans coalesce; the rebuilt VDS
   stays self-referencing (`'.'` → `data_000001`) and the retained link is
   trimmed normally. Do not let the two scans double-count again.

6. **Cross-version filter plugins: verbatim chunk copy needs `set_local`
   suppressed.** A filter's `set_local` stamps the *locally installed*
   plugin's version into `cd_values` at `H5Dcreate`. Real DECTRIS data was
   written by bitshuffle 0.3 (`cd_values=(0,3,2,0,2)`); the `hdf5plugin`
   here is 0.4, so `set_local` rewrites to `(0,4,2,0,3,2,0,2)` — which is
   *undecodable* (position 4 becomes 3=ZSTD, not 2=LZ4). No pre-image makes
   a newer plugin emit an older version, so `PipelineResolver` fails and it
   used to fall back to slow decode+recompress. Fix: `_preflight_pipelines`
   tests each compressed image pipeline once (resolver, filter registered);
   for any it cannot reproduce it calls `h5py.h5z.unregister_filter` so
   `set_local` cannot run, and `copy_frames` (`_create_verbatim`) then
   stores the source's `cd_values` verbatim and copies the compressed chunks
   byte-for-byte. Consequences to respect:
     - HDF5 refuses to create a data set with a *mandatory* unregistered
       filter, so the verbatim path marks the filter **optional**
       (`FLAG_OPTIONAL`). cd_values are byte-identical; only the flag
       differs. A reader lacking the plugin gets raw bytes instead of an
       error — but it could not read the source either.
     - Unregister only works with **no open data set using the filter**, so
       preflight must run before the walk opens any image data set.
     - Image data are the only thing these plugin filters compress here and
       we never decode them during the copy, so unregistering is safe. If a
       mask or `--also-compress` target ever used the same plugin filter,
       its decode would break — re-register around it if that ever happens.
     - `--verify` decodes, so `_reregister_filters` (`hdf5plugin.register()`)
       runs before it. Same-version cases let the resolver succeed and the
       mandatory path is kept untouched — do not route them through verbatim.

## Testing

    python nxmx_reduce.py -n 12 eiger/lyso_1_master.h5 -o out --verify -v

Cross-version check (catches item 2) — build a venv with an older HDF5 and
read the output back:

    python -m venv /tmp/oldenv
    /tmp/oldenv/bin/pip install "h5py==3.10.0" "numpy<2" hdf5plugin   # HDF5 1.14.2
    /tmp/oldenv/bin/python -c "import hdf5plugin,h5py; \
        print(h5py.File('out/lyso_1_data_000001.h5')['/entry/data/data'][0].sum())"

Topologies covered so far: VDS + parallel external links; external links
only (no VDS); plain contiguous stack in the master; self-referencing VDS
(`'.'`); self-referencing VDS whose source is an external link (real
DECTRIS filewriter, `ins10_1.nxs`); chunk[0] > frames kept; null
dataspaces; vlen-string and compound per-image arrays.

Verified end-to-end with real DIALS on `ins10_1.nxs` (3600 → 600):
`dials.import` geometry identical, `dials.find_spots` gives 27949 == 27949
reflections with zero per-frame mismatches and byte-identical centroids —
i.e. matches `dials.import image_range=1,600` on the original exactly. The
compressed chunks are now copied **verbatim** (see non-obvious item 6), not
decoded/recompressed: ~2 s for the copy and byte-identical to the source.

## Not yet verified

- Multi-module / multi-NXdata detectors (Jungfrau-style).
- Masters where a VDS source selection does not start at frame 0 —
  handled in code, never exercised.

## Style

Standard library + h5py + numpy only; `hdf5plugin` optional (import is
wrapped). Python 3.9+. No external build. Keep it a single file.
