# nxmx_reduce — project context

Tool that shrinks an NXmx data set: recompresses masks, truncates the image
stack to `-n` frames, and truncates every per-image data set to match.
Written for NE-CAT (APS 24-ID-C / 24-ID-E) Eiger2 data.

`--frames-per-data N` additionally *repartitions* the kept stack into output
data files of at most `N` frames each and rebuilds the virtual data set (and
the `data_00000i` links) to match — see item 7 below.

`--crop` cuts out the central area of an Eiger detector (16M→4M, 9M→1M by
default; `--crop-to CLASS` forces another target) and rewrites the geometry
header to match — see item 8 below.

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

7. **`--frames-per-data N` repartitions the stack — the output data files are
   decoupled from the source ones.** `_write_out_data_files` materialises the
   kept `nout` frames into fresh files of `N` frames each *before* the master
   walk, and `write_repartition_vds` (intercepted at the top of `copy_dataset`,
   before the `is_virtual`/`keeps` branches) rebuilds `entry/data/data` as a
   self-ref VDS over regenerated `data_00000i` external links. Source image
   files are "absorbed": their links are dropped in `visit` (matched against
   `absorbed_paths`) and they are never enqueued. Things that bit us / to keep:
     - **Only the DECTRIS self-ref-VDS-over-external-link layout is supported**
       (`plan.repartitionable`, set in `_scan_nxdata` when `resolve_extlink`
       fires). `_plan_repartition` rejects other layouts, multi-NXdata, mixed
       source dtype/shape/chunks/pipeline, and `chunks[0] != 1` — the verbatim
       per-frame copy (`_copy_one_frame`) needs one chunk per frame, and decode
       is unavailable anyway once preflight has unregistered the filter (item 6).
     - **A block may straddle a source boundary** (e.g. `-n 1500
       --frames-per-data 400` puts src1[800:1000]+src2[0:200] in one file).
       `frame_map[v] = (source, source_frame_index)` drives this; the uniform-
       pipeline precondition guarantees every source's chunks are byte-writable
       into the one output dataset.
     - **Naming continues the *source* pattern**, so filenames stay
       `ins10_1_00000i.h5` (retain original names). `_split_numbered` must strip
       the extension first — otherwise the `5` in `.h5` is taken as the counter
       and files come out `ins10_1_000004.h1`. `--verify` does *not* catch this
       (the VDS/links/files stay internally consistent); only inspecting the
       written filenames does.
     - VirtualSource shape is the block's real length (last block is the
       remainder), per item 3. `create_like`/`_create_verbatim` and the item-1
       assert are shared via `_create_image_like`. New files get `self.libver`
       (item 2). `verify` skips absorbed sources and relies on the master-stack
       readback (which decodes through the rebuilt VDS → links → new files).

8. **`--crop` cuts out the central detector area and *decodes* every frame — it
   is the one operation that cannot use the verbatim chunk copy.** A crop takes
   a sub-window of each frame, which is not a whole chunk, so the compressed
   bytes cannot be reused; frames are decoded, sliced, and recompressed
   (gzip+shuffle). Things that bit us / to keep:
     - **The module geometry is a hard-coded lookup over two Eiger families,
       keyed by image shape.** An Eiger image is a grid of modules separated by
       fixed gaps, and there are two geometries in the wild — told apart
       *unambiguously* by the image dimensions (no class in one family shares a
       shape with any class in the other):
         - **Eiger2** — `1028×512` (fast×slow) modules, `12 px` (fast) /
           `38 px` (slow) gaps → 16M = `4148×4362`. These were read off the gap
           pixels of a real Diamond I04 Eiger 16M (which, despite the
           `description = "Eiger 16M"`, uses this data-array geometry) and are
           verified end-to-end.
         - **Eiger (gen 1)** — `1030×514` modules, `10 px` (fast, "vertical
           join") / `37 px` (slow, "horizontal join") gaps → 16M = `4150×4371`.
       `EIGER_FAMILIES` holds both; `identify_eiger` returns
       `(family, class, grid)` for the matching `(slow, fast)` shape (classes
       16M=4×8, 9M=3×6, 4M=2×4, 1M=1×2, 500K=1×1), and `plan_crop` keeps the
       centred block of whole modules **using that family's pitch**, so the
       window always starts/ends on a module boundary and only discards whole
       outer modules and their gaps. Do not collapse the two families back to
       one set of constants — the Eiger2 real-data numbers (1028/512/12/38) and
       the Eiger1 nominal numbers (1030/514/10/37) are both correct, for
       different detectors.
     - **The crop is only valid if the header moves with it.** Three things
       change and DIALS reads all of them: image size (`x/y_pixels_in_detector`,
       `module/data_size` — which is `[slow, fast]`), the beam centre
       (`beam_center_x -= sx0`, `beam_center_y -= sy0`), and the detector origin
       (`module/module_offset`). The last is the one that matters for
       geometry: the module origin is displaced along the `fast/slow_pixel_
       direction` vectors by `(sx0, sy0)` pixels, then re-decomposed into
       magnitude+unit-vector. Cross-check: DIALS derives the beam centre purely
       from the `module_offset` chain and it must equal the written
       `beam_center_*` — the physical beam position is invariant under the crop
       (origin shift and beam-centre shift cancel). `data_origin` stays `[0,0]`.
     - **Preflight must NOT run when cropping** (it may unregister the filter
       for verbatim copy — item 6); crop needs the filter *present* to decode,
       so `_require_decodable` fails fast if a plugin is missing.
     - **Detector-sized 2D maps (masks, flatfields) are cropped too**, matched
       by trailing shape `== plan.image_shape`. Per-image 1D arrays (omega, …)
       are untouched.
     - Mutually exclusive with `--frames-per-data` (that path copies chunks
       verbatim; crop decodes) — combining them errors.
     - `--verify` *does* catch a bad crop: it compares each output frame with
       the correspondingly-cropped source frame (`_src_frame`).

## Testing

    python nxmx_reduce.py -n 12 eiger/lyso_1_master.h5 -o out --verify -v

Repartition (item 7) — verify covers divisible, remainder, and a block that
straddles a source boundary:

    python nxmx_reduce.py -n 600  --frames-per-data 100 ins10_1.nxs -o out --verify
    python nxmx_reduce.py -n 250  --frames-per-data 100 ins10_1.nxs -o out --verify  # 100,100,50
    python nxmx_reduce.py -n 1500 --frames-per-data 400 ins10_1.nxs -o out --verify  # block 3 crosses src edge

After repartitioning, *check the written filenames* (`ins10_1_00000i.h5`) and
the regenerated `data_00000i` links — `--verify` alone will not catch a naming
bug.

Crop (item 8) — `--verify` compares each frame with the cropped source; also
sanity-check the header:

    python nxmx_reduce.py -n 20 --crop ins10_1.nxs -o out --verify   # 16M -> 4M
    python nxmx_reduce.py -n 20 --crop-to 1M ins10_1.nxs -o out --verify

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

`--frames-per-data` verified end-to-end too: on `ins10_1.nxs` `-n 20
--frames-per-data 7` (original 4 files → 3 files of 7/7/6), `dials.import`
geometry is identical to the original and `dials.find_spots` gives 1046 ==
1046 reflections vs `image_range=1,20` on the original; a block that crosses a
source boundary (`-n 1500 --frames-per-data 400`) is byte-identical at the seam
(src1[999]==file3[199], src2[0]==file3[200]).

`--crop` verified end-to-end on `ins10_1.nxs` (Eiger 16M → 4M, `-n 20`):
`dials.import` reads the cropped panel as `2068×2162` px with beam centre
`(1081.07, 1121.33)` derived from the rewritten `module_offset` — exactly the
written `beam_center_*`. `dials.find_spots` finds 979 spots; the uncropped
20-frame reference finds 1046, of which exactly 979 lie inside the crop window,
and all 979 match the cropped spots within 1 px on the same frame (979/979).
The 67 spots on the discarded outer modules are correctly gone.

## Not yet verified

- Multi-module / multi-NXdata detectors (Jungfrau-style).
- Masters where a VDS source selection does not start at frame 0 —
  handled in code, never exercised.
- `--crop` on a **first-generation Eiger** (`1030×514`/`10`/`37`; e.g. 16M =
  `4150×4371`): the geometry and crop windows are derived and unit-tested
  (`plan_crop`/`identify_eiger`) but not yet run against a real Eiger1 file —
  the only real data on hand is the Eiger2-geometry `ins10_1.nxs`.
- `--crop` on layouts other than the self-ref-VDS-over-external-link one
  (contiguous-master / plain-external-link stacks) — the code path is generic
  but untested for cropping.

## Style

Standard library + h5py + numpy only; `hdf5plugin` optional (import is
wrapped). Python 3.9+. No external build. Keep it a single file.
