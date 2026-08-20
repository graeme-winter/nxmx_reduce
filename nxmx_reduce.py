#!/usr/bin/env python3
"""
nxmx_reduce.py -- shrink an NXmx data set.

Produces a self-consistent copy of an NXmx master file (plus any files it
links to) in an output directory, applying three reductions:

  1. mask data sets (``pixel_mask`` and friends), wherever they live -- in
     the master or in an externally linked file -- are rewritten with
     gzip + shuffle, which typically shrinks a 4M uint32 mask from ~16 MB
     to a few tens of kB;
  2. the image stack is truncated to the first ``-n`` images.  HDF5 virtual
     data sets are rebuilt against trimmed copies of only those source
     files that are still needed; source files that fall entirely beyond
     the cut are not copied at all and their links are dropped;
  3. any per-image data set reachable from the master -- goniometer axis
     arrays, per-frame count/frame times, timestamps, and so on, i.e. any
     data set whose slowest axis matches the original image count -- is
     truncated to match.

Image data are moved with ``read_direct_chunk``/``write_direct_chunk``, so
compressed frames are copied byte-for-byte without ever being decoded.
That means bitshuffle/LZ4 Eiger data can be reduced on a machine that has
no bitshuffle filter plugin installed, and no recompression cost is paid.
When the destination's exact filter recipe can be recreated the pipeline
is preserved unchanged; when it cannot (typically because the locally
installed filter plugin is a *different version* and its ``set_local``
callback rewrites ``cd_values`` differently) the chunks are still copied
verbatim, with the source's ``cd_values`` stored and the filter marked
optional -- see ``_preflight_pipelines`` / ``_create_verbatim``.  Only if
even that fails does it fall back to decode + recompress.

Examples
--------
    # keep the first 20 images
    nxmx_reduce.py -n 20 /data/lyso_1_master.h5 -o /tmp/lyso_small

    # keep 5 images, also recompress the flatfield, be chatty
    nxmx_reduce.py -n 5 master.nxs -o small --also-compress flatfield -v

    # keep 5 images and drop the flatfield correction entirely
    nxmx_reduce.py -n 5 master.nxs -o small --drop flatfield

    # keep 600 images, repartitioned into data files of 100 frames each
    nxmx_reduce.py -n 600 --frames-per-data 100 master.nxs -o small

    # keep 20 images and crop the central detector area (16M -> 4M, 9M -> 1M),
    # adjusting the header (image size, beam centre, module_offset)
    nxmx_reduce.py -n 20 --crop master.nxs -o small
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

import numpy as np

try:  # optional: lets us *read* bitshuffle data, and re-register a filter
    import hdf5plugin  # noqa: F401
    _HAVE_HDF5PLUGIN = True
except ImportError:
    hdf5plugin = None
    _HAVE_HDF5PLUGIN = False

import h5py

__version__ = "1.3"

# Data set base names treated as masks (recompressed rather than copied).
DEFAULT_MASK_NAMES = ("pixel_mask", "mask", "*_mask")

# Scalar book-keeping data sets that describe the length of the series and
# so need patching when the series is truncated.
NIMAGE_SCALARS = ("nimages", "num_images", "number_of_images")


# --------------------------------------------------------------------------
# Eiger detector module geometry and centred cropping (--crop)
# --------------------------------------------------------------------------
#
# An Eiger image is a rectangular grid of identical sensor modules separated by
# blank inter-module gaps.  There are two module geometries in the wild, and
# they are told apart *unambiguously* by the image dimensions (no class in one
# family shares a shape with any class in the other):
#
#   * Eiger (first generation): module 1030 (fast) x 514 (slow) px, with a
#     10 px (fast / "vertical join") / 37 px (slow / "horizontal join") gap.
#     16M -> 4150 x 4371 (fast x slow).
#   * Eiger2: module 1028 x 512 px, with a 12 px (fast) / 38 px (slow) gap.
#     16M -> 4148 x 4362.  These constants were read straight off the gap
#     pixels of a real Diamond I04 Eiger 16M and are verified end-to-end.
#
# Knowing the module pitch lets a crop land exactly on a module boundary, so it
# only ever discards whole outer modules and their adjacent gaps.
EIGER_FAMILIES = (
    dict(name="Eiger2", module_fast=1028, module_slow=512,
         gap_fast=12, gap_slow=38),
    dict(name="Eiger", module_fast=1030, module_slow=514,
         gap_fast=10, gap_slow=37),
)

# Named Eiger classes as (modules_fast, modules_slow) grids (shared by both
# families).  Applying a family's module/gap sizes gives its data-array shape,
# e.g. Eiger2 16M -> 4148 x 4362, Eiger 16M -> 4150 x 4371 (fast x slow).
EIGER_CLASS_GRID = {
    "500K": (1, 1),
    "1M": (1, 2),
    "4M": (2, 4),
    "9M": (3, 6),
    "16M": (4, 8),
}

# Default central crop target for each source class (as the user asked:
# 16M -> 4M, 9M -> 1M).  A different target may be forced with --crop-to.
EIGER_DEFAULT_CROP = {"16M": "4M", "9M": "1M"}


def _grid_dims(fam, grid):
    """(fast, slow) data-array pixel dimensions of a grid in family ``fam``."""
    nf, ns = grid
    fast = nf * fam["module_fast"] + (nf - 1) * fam["gap_fast"]
    slow = ns * fam["module_slow"] + (ns - 1) * fam["gap_slow"]
    return fast, slow


def identify_eiger(image_shape):
    """Identify the Eiger detector whose data array is ``image_shape`` = (slow,
    fast).

    Returns ``(family, class_name, grid)`` or ``(None, None, None)`` if no
    class in any family matches.
    """
    slow, fast = int(image_shape[0]), int(image_shape[1])
    for fam in EIGER_FAMILIES:
        for name, grid in EIGER_CLASS_GRID.items():
            if _grid_dims(fam, grid) == (fast, slow):
                return fam, name, grid
    return None, None, None


def plan_crop(image_shape, target=None):
    """Work out the centred pixel window to crop ``image_shape`` to ``target``.

    ``image_shape`` is (slow, fast).  ``target`` is an Eiger class name
    (e.g. ``'4M'``) or ``None`` to use ``EIGER_DEFAULT_CROP``.  The detector
    family (module/gap geometry) is inferred from ``image_shape`` and the crop
    stays within that family.  Returns ``(sy0, sy1, sx0, sx1, info)`` -- the
    window is ``[sy0:sy1, sx0:sx1]`` on a (slow, fast) frame and always
    begins/ends on a module boundary, so only whole outer modules (and the gaps
    beside them) are discarded.  Raises ``SystemExit`` if the source is not a
    recognised Eiger class or the target does not fit inside it.
    """
    fam, name, grid = identify_eiger(image_shape)
    if fam is None:
        known = ", ".join(
            f"{f['name']} {n} {_grid_dims(f, g)[0]}x{_grid_dims(f, g)[1]}"
            for f in EIGER_FAMILIES for n, g in EIGER_CLASS_GRID.items())
        raise SystemExit(
            f"--crop: image shape (slow,fast)={tuple(image_shape)} is not a "
            "recognised Eiger module grid; cannot work out the module layout. "
            f"Known (fast x slow): {known}")
    if target is None:
        target = EIGER_DEFAULT_CROP.get(name)
        if target is None:
            raise SystemExit(
                f"--crop: no default crop target for an {fam['name']} {name}; "
                f"pass one with --crop-to (choices: {', '.join(EIGER_CLASS_GRID)})")
    target = target.upper()
    if target not in EIGER_CLASS_GRID:
        raise SystemExit(
            f"--crop: unknown target {target!r}; choose one of "
            f"{', '.join(EIGER_CLASS_GRID)}")
    nf, ns = grid
    tf, ts = EIGER_CLASS_GRID[target]
    if tf > nf or ts > ns:
        raise SystemExit(
            f"--crop: cannot crop an {fam['name']} {name} ({nf}x{ns} modules) "
            f"to {target} ({tf}x{ts} modules) -- the target is larger")
    # centred block of whole modules
    f0, s0 = (nf - tf) // 2, (ns - ts) // 2
    pitch_f = fam["module_fast"] + fam["gap_fast"]
    pitch_s = fam["module_slow"] + fam["gap_slow"]
    sx0 = f0 * pitch_f
    sx1 = sx0 + tf * fam["module_fast"] + (tf - 1) * fam["gap_fast"]
    sy0 = s0 * pitch_s
    sy1 = sy0 + ts * fam["module_slow"] + (ts - 1) * fam["gap_slow"]
    info = (f"{fam['name']} {name} ({nf}x{ns}) -> {target} ({tf}x{ts}); keep "
            f"modules fast[{f0}:{f0 + tf}] slow[{s0}:{s0 + ts}], "
            f"window slow[{sy0}:{sy1}] fast[{sx0}:{sx1}] "
            f"-> ({sy1 - sy0}, {sx1 - sx0})")
    return sy0, sy1, sx0, sx1, info


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def human(nbytes: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes:.0f} B"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def _split_numbered(name: str):
    """Split a name around its last run of digits.

    ``'data_000001'``        -> ``('data_', 6, '')``
    ``'ins10_1_000001.h5'``  -> ``('ins10_1_', 6, '.h5')``

    so ``f"{pre}{i:0{width}d}{suf}"`` regenerates the sequence.  Numbers
    inside the stem (``ins10_1``) are left alone -- only the final digit run,
    which is the per-file index, is treated as the counter.
    """
    # strip the extension first so a digit in it (the '5' of '.h5') is not
    # mistaken for the counter
    stem, ext = os.path.splitext(name)
    m = re.search(r"(\d+)(\D*)$", stem)
    if not m:
        # no trailing number to continue; fall back to a plain suffix counter
        return (stem + "_", 6, ext)
    prefix = stem[:m.start(1)]
    return (prefix, len(m.group(1)), m.group(2) + ext)


def copy_attrs(src, dst) -> None:
    """Copy attributes, preserving dtype/shape where h5py lets us."""
    for key in src.attrs:
        value = src.attrs[key]
        try:
            aid = src.attrs.get_id(key)
            dst.attrs.create(key, value, shape=aid.shape, dtype=aid.dtype)
        except Exception:
            try:
                dst.attrs[key] = value
            except Exception as exc:  # pragma: no cover - pathological attrs
                print(f"  ! could not copy attribute {key!r}: {exc}",
                      file=sys.stderr)


def is_virtual(dset: h5py.Dataset) -> bool:
    try:
        return dset.is_virtual
    except Exception:
        return False


def iter_chunk_offsets(shape, chunks, nframes):
    """Yield chunk origins covering frames [0, nframes) of a chunked set."""
    extents = (nframes,) + tuple(shape[1:])
    ranges = [range(0, max(e, 1), c) for e, c in zip(extents, chunks)]

    def rec(dim, prefix):
        if dim == len(ranges):
            yield tuple(prefix)
            return
        for off in ranges[dim]:
            yield from rec(dim + 1, prefix + [off])

    if all(len(r) for r in ranges):
        yield from rec(0, [])


def filters_of(dcpl) -> tuple:
    """The filter pipeline of a DCPL as ((code, flags, cd_values), ...)."""
    out = []
    for i in range(dcpl.get_nfilters()):
        code, flags, values, _name = dcpl.get_filter(i)
        out.append((int(code), int(flags), tuple(int(v) for v in values)))
    return tuple(out)


def build_dcpl(chunks, filters, fillvalue, dtype):
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk(tuple(chunks))
    if fillvalue is not None:
        try:
            dcpl.set_fill_value(np.array(fillvalue, dtype=dtype))
        except Exception:
            pass
    for code, flags, values in filters:
        dcpl.set_filter(code, flags, values)
    return dcpl


class PipelineResolver:
    """Work out how to recreate a data set's exact filter pipeline.

    This is subtler than it looks.  A filter may register a ``set_local``
    callback that rewrites ``cd_values`` at H5Dcreate time -- bitshuffle,
    for instance, splices in the element size.  So the values *stored in
    the file* are not the values you must *pass* to recreate it: hand the
    stored ones back to H5Dcreate and set_local mangles them a second
    time, giving a data set whose chunks are byte-perfect but whose
    recorded pipeline no longer decodes (bslz4 then fails with "Non
    integer number of elements").

    There is no API to ask a filter what its pre-set_local input was, so
    we search for it: create throwaway data sets in an in-memory file
    until one round-trips to exactly the pipeline we are trying to
    reproduce.  Results are cached per (dtype, chunk, pipeline).
    """

    MAX_PROBES = 256

    def __init__(self, verbose=False):
        self.cache: dict = {}
        self.verbose = verbose

    @staticmethod
    def _variants(target):
        """Candidate inputs: the stored values, then values with short runs
        removed (set_local generally *inserts*, so the pre-image is
        shorter)."""
        yield target
        for j, (code, flags, values) in enumerate(target):
            for k in (1, 2, 3):
                for p in range(len(values) - k + 1):
                    trimmed = values[:p] + values[p + k:]
                    yield target[:j] + ((code, flags, trimmed),) + target[j + 1:]

    def _probe(self, filters, chunks, fillvalue, dtype, tid):
        """Create a scratch data set and report the pipeline it ends up with."""
        try:
            dcpl = build_dcpl(chunks, filters, fillvalue, dtype)
        except Exception:
            return None
        try:
            with h5py.File("probe", "w", driver="core",
                           backing_store=False) as fh:
                space = h5py.h5s.create_simple(tuple(chunks), tuple(chunks))
                dsid = h5py.h5d.create(fh.id, b"probe", tid, space, dcpl=dcpl)
                return filters_of(dsid.get_create_plist())
        except Exception:
            return None

    def resolve(self, src: h5py.Dataset):
        """Filter list to pass to H5Dcreate, or None if we cannot match."""
        target = filters_of(src.id.get_create_plist())
        chunks = tuple(src.chunks)
        key = (src.dtype.str, chunks, target)
        if key in self.cache:
            return self.cache[key]
        tid = src.id.get_type()
        result = None
        for i, cand in enumerate(self._variants(target)):
            if i >= self.MAX_PROBES:
                break
            if self._probe(cand, chunks, src.fillvalue, src.dtype,
                           tid) == target:
                result = cand
                if i and self.verbose:
                    print(f"  . filter pipeline for {src.name}: set_local "
                          f"rewrites cd_values; using pre-image {cand}")
                break
        self.cache[key] = result
        return result


def create_like(parent: h5py.Group, name: str, src: h5py.Dataset, shape,
                filters):
    """Create a data set with the source's exact type and filter pipeline."""
    chunks = src.chunks
    maxshape = tuple(max(s, c) for s, c in zip(shape, chunks))
    dcpl = build_dcpl(chunks, filters, src.fillvalue, src.dtype)
    space = h5py.h5s.create_simple(tuple(shape), maxshape)
    dsid = h5py.h5d.create(parent.id, name.encode("utf-8"),
                           src.id.get_type(), space, dcpl=dcpl)
    return h5py.Dataset(dsid)


# --------------------------------------------------------------------------
# planning: work out where the images live and how many to keep from each
# --------------------------------------------------------------------------


class ImageSource:
    """One file+path holding part of the image stack."""

    __slots__ = ("path", "dset", "length", "keep", "vstart", "sstart",
                 "rvstart", "rused")

    def __init__(self, path: Path, dset: str, length: int):
        self.path = path
        self.dset = dset
        self.length = length
        self.keep = 0
        self.vstart = None      # offset in the virtual (stitched) stack
        self.sstart = 0         # first frame used from this source
        self.rvstart = 0        # resolved virtual start (filled by Plan.apply)
        self.rused = 0          # resolved number of frames used (Plan.apply)

    @property
    def key(self):
        return (str(self.path), self.dset)


def dataset_length(path: Path, dset: str):
    try:
        with h5py.File(path, "r") as fh:
            obj = fh.get(dset)
            if isinstance(obj, h5py.Dataset) and obj.ndim:
                return int(obj.shape[0])
    except Exception:
        pass
    return None


def resolve_extlink(container: h5py.File, dset_name: str, here: Path):
    """Follow one external-link hop for a name inside `container`.

    In the DECTRIS filewriter topology the image VDS is *self-referencing*
    (source file ``'.'``) and its ``dset_name`` names an external link in
    the master -- e.g. ``/entry/data/data_000001`` -- rather than a real
    data set.  That link points at ``/data`` in a separate data file.  To
    plan and rebuild such a VDS we must see through the link to the real
    backing store.

    Returns ``(real_path, real_dset)`` if `dset_name` is an external link,
    else ``(None, None)``.
    """
    try:
        link = container.get(dset_name, getlink=True)
    except Exception:
        return None, None
    if isinstance(link, h5py.ExternalLink):
        tgt = Path(link.filename)
        if not tgt.is_absolute():
            tgt = here / tgt
        return tgt.resolve(), link.path
    return None, None


def find_nxdata_groups(fh: h5py.File):
    """NXdata groups, by class attribute, with /entry/data as a fallback."""
    found = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            cls = obj.attrs.get("NX_class", b"")
            if isinstance(cls, bytes):
                cls = cls.decode("utf-8", "replace")
            if cls == "NXdata":
                found.append(obj)

    fh.visititems(visitor)
    if not found and "/entry/data" in fh:
        found.append(fh["/entry/data"])
    return found


def span_of(space) -> tuple:
    """(start, length) along the slowest axis of a dataspace selection.

    Handles the H5S_SEL_ALL / H5S_SEL_NONE cases, for which
    ``get_select_bounds`` is not defined.
    """
    try:
        sel = space.get_select_type()
    except Exception:
        sel = None
    if sel == h5py.h5s.SEL_NONE:
        return 0, 0
    if sel == h5py.h5s.SEL_ALL:
        dims = space.get_simple_extent_dims()
        return 0, int(dims[0]) if dims else 0
    try:
        lo, hi = space.get_select_bounds()
        return int(lo[0]), int(hi[0]) - int(lo[0]) + 1
    except (ValueError, RuntimeError):
        dims = space.get_simple_extent_dims()
        return 0, int(dims[0]) if dims else 0


class Plan:
    """Which datasets hold images, how long they are, how much to keep."""

    def __init__(self, master: Path, verbose=False):
        self.master = master.resolve()
        self.verbose = verbose
        self.sources: dict[tuple, ImageSource] = {}
        self.stack_paths: set[str] = set()   # paths of image stacks in master
        self.nimages = 0
        self.image_shape = None
        # repartition template: filled when a self-ref-VDS-over-external-link
        # stack is found (the DECTRIS filewriter layout).  Lets --frames-per-data
        # regenerate the same structure with a different number of data files.
        self.repartitionable = False
        self.stack_vds_path = None       # master path of the image VDS
        self.nxdata_group_path = None    # group holding it, e.g. /entry/data
        self.data_internal_path = None   # dataset path inside each data file
        self.link_name_example = None    # e.g. 'data_000001'
        self.file_name_example = None    # e.g. 'ins10_1_000001.h5'
        self._build()

    def _add(self, path: Path, dset: str, length):
        src = self.sources.get((str(path), dset))
        if src is None:
            if not path.exists():
                # e.g. an unlimited printf-pattern VDS mapping, or a file
                # that was never written -- nothing to copy or map
                print(f"  ! image source {path} does not exist; ignoring",
                      file=sys.stderr)
                return None
            if length is None:
                length = dataset_length(path, dset)
            if length is None:
                return None
            src = ImageSource(path, dset, int(length))
            self.sources[src.key] = src
        return src

    def _build(self):
        with h5py.File(self.master, "r") as fh:
            for grp in find_nxdata_groups(fh):
                self._scan_nxdata(fh, grp)
        if not self.nimages:
            raise SystemExit(
                f"{self.master}: could not locate an image data set "
                "(no NXdata group with a >=3D data set was found)")

    def _scan_nxdata(self, fh, grp):
        here = self.master.parent

        # 1. a real or virtual stack sitting in the master itself
        for name in list(grp):
            link = grp.get(name, getlink=True)
            if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                continue
            obj = grp.get(name)
            if not isinstance(obj, h5py.Dataset) or obj.ndim < 3:
                continue
            self.nimages = max(self.nimages, int(obj.shape[0]))
            self.image_shape = tuple(obj.shape[1:])
            self.stack_paths.add(obj.name)
            if is_virtual(obj):
                for vmap in obj.virtual_sources():
                    fname = vmap.file_name
                    selfref = fname in (".", "")
                    tgt = (self.master if selfref
                           else (here / fname).resolve())
                    dpath = vmap.dset_name
                    if selfref:
                        # self-ref source may name an external link, not a
                        # real data set (DECTRIS filewriter layout); key the
                        # plan against the real backing file so it is not
                        # double-counted with the direct external-link scan
                        real_path, real_dpath = resolve_extlink(fh, dpath, here)
                        if real_path is not None:
                            # this is the layout --frames-per-data can rebuild:
                            # remember how to regenerate the link/file/VDS names
                            self.repartitionable = True
                            self.stack_vds_path = obj.name
                            self.nxdata_group_path = grp.name
                            self.data_internal_path = real_dpath
                            self.link_name_example = dpath.rsplit("/", 1)[-1]
                            self.file_name_example = real_path.name
                            tgt, dpath = real_path, real_dpath
                    vstart, vlen = span_of(vmap.vspace)
                    sstart, slen = span_of(vmap.src_space)
                    length = dataset_length(tgt, dpath)
                    if length is None:
                        # unreadable source; trust the mapping extent
                        length = sstart + min(vlen, slen)
                    src = self._add(tgt, dpath, length)
                    if src is not None:
                        src.vstart = vstart          # where it lands in the stack
                        src.sstart = sstart
            else:
                self._add(self.master, obj.name, obj.shape[0])

        # 2. classic per-file external links: data_000001, data_000002, ...
        links = []
        for name in sorted(grp):
            link = grp.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                tgt = Path(link.filename)
                if not tgt.is_absolute():
                    tgt = (here / tgt)
                links.append((tgt.resolve(), link.path))
        offset = 0
        for tgt, dpath in links:
            length = dataset_length(tgt, dpath)
            if length is None:
                continue
            self._add(tgt, dpath, length)
            offset += length
        if not self.nimages:
            self.nimages = offset

    # -- once nimages_out is known ------------------------------------
    def apply(self, nout: int):
        """Decide how many frames to keep from each source."""
        # Ordering: VDS mappings carry an explicit virtual offset; plain
        # external links are laid out in name order.
        ordered = sorted(self.sources.values(),
                         key=lambda s: (s.vstart is None,
                                        (s.vstart or 0), s.dset))
        running = 0
        for src in ordered:
            vstart = src.vstart
            sstart = src.sstart
            if vstart is None:
                vstart = running
            used = max(0, min(src.length - sstart, nout - vstart))
            src.keep = max(src.keep, min(src.length, sstart + used))
            src.rvstart = vstart     # resolved position in the output stack
            src.rused = used         # frames this source contributes
            running = vstart + (src.length - sstart)

    def segments(self):
        """Sources contributing at least one frame, in output-stack order."""
        segs = [s for s in self.sources.values() if s.rused > 0]
        return sorted(segs, key=lambda s: s.rvstart)

    def for_file(self, path: Path):
        return {k[1]: v for k, v in self.sources.items() if k[0] == str(path)}

    def file_is_empty(self, path: Path) -> bool:
        entries = self.for_file(path)
        return bool(entries) and all(s.keep == 0 for s in entries.values())


# --------------------------------------------------------------------------
# the reducer
# --------------------------------------------------------------------------


class Reducer:
    def __init__(self, args):
        self.args = args
        self.master = Path(args.master).resolve()
        self.outdir = Path(args.output).resolve()
        self.plan = Plan(self.master, args.verbose)
        self.nin = self.plan.nimages
        self.nout = min(args.num_images, self.nin) if args.num_images else self.nin
        self.plan.apply(self.nout)
        self.mask_patterns = list(DEFAULT_MASK_NAMES) + list(args.also_compress)
        self.drop_patterns = list(args.drop)
        self.libver = ("earliest", args.libver)
        self.pipelines = PipelineResolver(args.verbose)
        self.outnames: dict[str, str] = {}     # resolved src path -> out name
        self.queue: list[Path] = []
        self.done: set[str] = set()
        self.stats = {"masks": 0, "trimmed": 0, "frames": 0, "files": 0,
                      "dropped": 0}
        # filters we unregistered so their chunks copy verbatim (see
        # _preflight_pipelines); re-registered before --verify
        self.verbatim_filters: set[int] = set()

        # -- repartition (--frames-per-data) ---------------------------
        self.repartition = args.frames_per_data
        self.absorbed_paths: set[Path] = set()
        self.repart_files: list[tuple] = []      # (lo, hi, filename, linkname)
        self.reserved_names: set[str] = set()    # output basenames not to reuse
        if self.repartition:
            self._plan_repartition()

        # -- central crop (--crop) -------------------------------------
        # crop=(sy0, sy1, sx0, sx1) window on a (slow, fast) frame, or None.
        # Cropping decodes every frame (a sub-window is not a whole chunk), so
        # it is incompatible with the verbatim chunk-copy fast path and with
        # --frames-per-data's verbatim repartition.
        self.crop = None
        self.crop_shape = None                   # (new_slow, new_fast)
        self.crop_info = None
        if args.crop or args.crop_to:
            if self.repartition:
                raise SystemExit(
                    "--crop cannot currently be combined with "
                    "--frames-per-data (crop must decode frames; repartition "
                    "copies chunks verbatim)")
            if self.plan.image_shape is None:
                raise SystemExit("--crop: could not determine the image shape")
            sy0, sy1, sx0, sx1, info = plan_crop(self.plan.image_shape,
                                                 args.crop_to)
            self.crop = (sy0, sy1, sx0, sx1)
            self.crop_shape = (sy1 - sy0, sx1 - sx0)
            self.crop_info = info

    # -- repartition planning ------------------------------------------
    def _plan_repartition(self):
        """Validate the layout and lay out the new data files/links."""
        if self.repartition < 1:
            raise SystemExit("--frames-per-data must be at least 1")
        if not self.plan.repartitionable:
            raise SystemExit(
                "--frames-per-data currently supports the DECTRIS "
                "self-referencing-VDS-over-external-link layout (master VDS "
                "'entry/data/data' -> 'data_00000N' external links -> data "
                "files); this data set does not use it, so its image "
                "partitioning cannot be rebuilt.  Omit --frames-per-data to "
                "reduce it with the source partitioning preserved.")
        if len(self.plan.stack_paths) != 1:
            raise SystemExit(
                "--frames-per-data supports a single image stack; this data "
                f"set has {len(self.plan.stack_paths)}.")
        segs = self.plan.segments()
        if not segs:
            raise SystemExit("--frames-per-data: no image frames to repartition")
        # every source feeding the stack must share one geometry + pipeline so
        # a single output data set can hold any frame and take its compressed
        # chunks verbatim (write_direct_chunk stores raw bytes against the
        # destination's own recipe -- a mismatched source would be undecodable)
        ref = None
        with_open = {}
        try:
            for s in segs:
                fh = with_open.get(str(s.path))
                if fh is None:
                    fh = with_open[str(s.path)] = h5py.File(s.path, "r")
                d = fh[s.dset]
                spec = (d.dtype.str, tuple(d.shape[1:]), tuple(d.chunks or ()),
                        filters_of(d.id.get_create_plist()))
                if ref is None:
                    ref = spec
                elif spec != ref:
                    raise SystemExit(
                        "--frames-per-data: image sources differ in "
                        "dtype/shape/chunks/compression; cannot repartition "
                        "into a single uniform data set.")
            if not ref[2] or ref[2][0] != 1:
                raise SystemExit(
                    "--frames-per-data currently requires image data chunked "
                    "one frame per chunk (chunks[0] == 1); this data set is "
                    f"chunked {ref[2]}.")
        finally:
            for fh in with_open.values():
                fh.close()

        # continue the source's own naming so filenames stay e.g.
        # ins10_1_00000i.h5 with data_00000i links (identical to the originals
        # when the frame counts line up)
        file_pre, file_w, file_suf = _split_numbered(self.plan.file_name_example)
        link_pre, link_w, link_suf = _split_numbered(self.plan.link_name_example)
        reserved = {self.outname_for(self.master)}
        K = self.repartition
        for i, lo in enumerate(range(0, self.nout, K), start=1):
            hi = min(lo + K, self.nout)
            fname = f"{file_pre}{i:0{file_w}d}{file_suf}"
            lname = f"{link_pre}{i:0{link_w}d}{link_suf}"
            if fname in reserved:
                raise SystemExit(
                    f"--frames-per-data: generated data-file name {fname!r} "
                    "collides with another output file")
            reserved.add(fname)
            self.repart_files.append((lo, hi, fname, lname))
        # keep the walk's outname_for from later handing a copied file one of
        # these names and overwriting a data file we already wrote
        self.reserved_names = {f for (_, _, f, _) in self.repart_files}
        self.absorbed_paths = {s.path for s in self.plan.sources.values()}

    # -- crop helpers --------------------------------------------------
    def _is_detector_2d(self, src) -> bool:
        """True if `src`'s trailing two axes span the full (uncropped) frame."""
        return (self.crop is not None and getattr(src, "ndim", 0) >= 2
                and src.shape is not None
                and tuple(src.shape[-2:]) == tuple(self.plan.image_shape))

    def _crop2(self, arr):
        """Crop the trailing (slow, fast) axes of an array to the window."""
        sy0, sy1, sx0, sx1 = self.crop
        return arr[..., sy0:sy1, sx0:sx1]

    def _src_frame(self, dset, i):
        """Frame ``i`` of a source stack, cropped to the window if cropping."""
        if self.crop is not None:
            sy0, sy1, sx0, sx1 = self.crop
            return dset[i, sy0:sy1, sx0:sx1]
        return dset[i]

    def _img_shape(self, src) -> tuple:
        """Per-frame shape of an image data set after any crop."""
        if self.crop is not None:
            return tuple(self.crop_shape)
        return tuple(src.shape[1:])

    def _crop_compression(self) -> dict:
        """``create_dataset`` compression kwargs for recompressed cropped data.

        A crop decodes each frame and recompresses it, so we re-emit the native
        Eiger pipeline -- bitshuffle + LZ4 (bslz4) -- rather than gzip+shuffle.
        ``--crop`` already requires ``hdf5plugin`` to decode the source (see
        ``_require_decodable``), so the filter is present; fall back to
        gzip+shuffle only in the unexpected case that it is not."""
        if _HAVE_HDF5PLUGIN:
            return dict(hdf5plugin.Bitshuffle(cname="lz4"))
        return dict(shuffle=True, compression="gzip",
                    compression_opts=self.args.compression_level)

    def _require_decodable(self):
        """--crop must decode frames; make sure every image filter is present.

        Unlike the verbatim copy path, cropping reads and slices each frame, so
        a missing compression plugin would fail mid-copy.  Fail fast instead.
        """
        for src in self.plan.sources.values():
            if src.keep == 0:
                continue
            try:
                fh = h5py.File(src.path, "r")
            except OSError:
                continue
            with fh:
                obj = fh.get(src.dset)
                if not isinstance(obj, h5py.Dataset) or not obj.chunks:
                    continue
                for code, _flags, _values in filters_of(obj.id.get_create_plist()):
                    if (code >= h5py.h5z.FILTER_RESERVED
                            and not h5py.h5z.filter_avail(code)):
                        raise SystemExit(
                            f"--crop must decode image frames, but the "
                            f"compression filter {code} used by "
                            f"{src.path.name} is not available here; install "
                            "the plugin (e.g. pip install hdf5plugin) and retry")

    # -- naming --------------------------------------------------------
    def outname_for(self, path: Path) -> str:
        key = str(path)
        if key in self.outnames:
            return self.outnames[key]
        base = path.name
        stem, ext = os.path.splitext(base)
        n = 1
        while base in self.outnames.values() or base in self.reserved_names:
            base = f"{stem}_{n}{ext}"
            n += 1
        self.outnames[key] = base
        return base

    def enqueue(self, path: Path) -> str:
        name = self.outname_for(path)
        if str(path) not in self.done and path not in self.queue:
            self.queue.append(path)
        return name

    # -- classification ------------------------------------------------
    def is_mask(self, name: str, dset: h5py.Dataset) -> bool:
        if dset.ndim < 2 or dset.dtype.kind not in "uifb":
            return False
        return any(fnmatch.fnmatch(name, pat) for pat in self.mask_patterns)

    def is_dropped(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, pat) for pat in self.drop_patterns)

    def is_per_image(self, dset: h5py.Dataset) -> bool:
        if not dset.ndim or dset.shape[0] != self.nin or self.nin < 2:
            return False
        # don't mistake a detector-sized image (e.g. a flatfield whose slow
        # axis happens to equal the frame count) for a per-image array
        if self.plan.image_shape and tuple(dset.shape) == self.plan.image_shape:
            return False
        return True

    # -- driving -------------------------------------------------------
    def run(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        if self.outdir == self.master.parent:
            raise SystemExit("refusing to write output into the input "
                             "directory -- choose a different -o")
        if self.crop is not None:
            # crop decodes frames, so the verbatim-copy preflight (which may
            # unregister the filter) must not run; instead insist the filter is
            # present so the decode cannot fail half way through
            self._require_decodable()
            if self.args.verbose:
                print(f"[crop] {self.crop_info}")
        else:
            self._preflight_pipelines()
        if self.repartition:
            self._write_out_data_files()
        self.enqueue(self.master)
        while self.queue:
            src_path = self.queue.pop(0)
            if str(src_path) in self.done:
                continue
            self.done.add(str(src_path))
            self.process_file(src_path)
        rc = self.report()
        if self.args.verify:
            self._reregister_filters()
            rc = self.verify() or rc
        return rc

    # -- filter-pipeline preflight -------------------------------------
    def _preflight_pipelines(self):
        """Decide, per compressed image pipeline, whether its exact filter
        recipe can be recreated at ``H5Dcreate`` time.

        A filter's ``set_local`` callback rewrites ``cd_values`` at create
        time.  When the locally installed plugin is a *different version*
        from the one that wrote the file, ``set_local`` stamps its own
        version into ``cd_values`` and there is no input that reproduces
        the stored recipe (``PipelineResolver`` searches and fails).  The
        compressed chunks are still byte-identical, though, so rather than
        decode + recompress we copy them verbatim -- which needs the
        destination created with the source's ``cd_values`` unchanged.  The
        only way to stop ``set_local`` from meddling is to unregister the
        filter, so we do, here, before any image data set is opened (HDF5
        refuses to unregister a filter that an open data set is using).

        Image data are the only thing these plugin filters compress and we
        never decode them during the copy, so unregistering is safe.  The
        filter is re-registered before ``--verify`` reads frames back.
        """
        seen: set = set()
        for src in self.plan.sources.values():
            if src.keep == 0:
                continue
            try:
                fh = h5py.File(src.path, "r")
            except OSError:
                continue
            with fh:
                obj = fh.get(src.dset)
                if not isinstance(obj, h5py.Dataset) or not obj.chunks:
                    continue
                pipe = filters_of(obj.id.get_create_plist())
                if not pipe or pipe in seen:
                    continue
                seen.add(pipe)
                reproducible = self.pipelines.resolve(obj) is not None
            if reproducible:
                continue
            for code, _flags, _values in pipe:
                if code < h5py.h5z.FILTER_RESERVED:
                    continue           # built-in (deflate/shuffle/...) -- leave it
                try:
                    if h5py.h5z.filter_avail(code):
                        h5py.h5z.unregister_filter(code)
                    self.verbatim_filters.add(int(code))
                    if self.args.verbose:
                        print(f"  . filter {code}: cannot reproduce its "
                              "cd_values via set_local (version mismatch?); "
                              "will copy chunks verbatim, marked optional")
                except Exception as exc:
                    print(f"  ! could not unregister filter {code} "
                          f"({exc}); may fall back to recompress",
                          file=sys.stderr)

    def _reregister_filters(self):
        """Re-register any filter unregistered in preflight, so --verify can
        decode frames again."""
        if not self.verbatim_filters:
            return
        if _HAVE_HDF5PLUGIN:
            try:
                hdf5plugin.register()
            except Exception as exc:  # pragma: no cover
                print(f"  ! could not re-register filter plugins for verify "
                      f"({exc}); frame read-back may fail", file=sys.stderr)

    def process_file(self, src_path: Path):
        out_path = self.outdir / self.outname_for(src_path)
        if self.args.verbose:
            print(f"[file] {src_path} -> {out_path}")
        try:
            fin = h5py.File(src_path, "r")
        except OSError as exc:
            print(f"  ! cannot open {src_path}: {exc}", file=sys.stderr)
            return
        keeps = self.plan.for_file(src_path)
        # NB libver: "latest" on HDF5 >= 2.0 emits v5 chunked-layout messages
        # that HDF5 1.14 and earlier cannot parse ("bad version number for
        # layout message"), which breaks DIALS/XDS.  v110 is the oldest bound
        # that still supports virtual data sets.
        with fin, h5py.File(out_path, "w", libver=self.libver) as fout:
            copy_attrs(fin, fout)
            self.visit(fin, fout, src_path.parent, keeps)
        self.stats["files"] += 1

    def visit(self, sgrp, dgrp, srcdir: Path, keeps: dict):
        copy_attrs(sgrp, dgrp)
        for name in sgrp:
            link = sgrp.get(name, getlink=True)

            if isinstance(link, h5py.SoftLink):
                dgrp[name] = h5py.SoftLink(link.path)
                continue

            if isinstance(link, h5py.ExternalLink):
                tgt = Path(link.filename)
                if not tgt.is_absolute():
                    tgt = srcdir / tgt
                tgt = tgt.resolve()
                if self.repartition and tgt in self.absorbed_paths:
                    # this image source is being repartitioned; its link is
                    # regenerated (renumbered) in write_repartition_vds
                    if self.args.verbose:
                        print(f"  - repartition: dropping source link {name} "
                              f"(-> {tgt.name})")
                    continue
                if self.plan.file_is_empty(tgt):
                    if self.args.verbose:
                        print(f"  - dropping link {name} "
                              f"(-> {tgt.name}, no frames retained)")
                    continue
                if not tgt.exists():
                    print(f"  ! missing external file {tgt}, link {name} "
                          "copied verbatim", file=sys.stderr)
                    dgrp[name] = h5py.ExternalLink(link.filename, link.path)
                    continue
                dgrp[name] = h5py.ExternalLink(self.enqueue(tgt), link.path)
                continue

            obj = sgrp[name]
            if isinstance(obj, h5py.Group):
                sub = dgrp.create_group(name)
                self.visit(obj, sub, srcdir, keeps)
            elif isinstance(obj, h5py.Dataset):
                self.copy_dataset(name, obj, dgrp, srcdir, keeps)
            else:  # named datatype
                dgrp[name] = obj

    # -- data set dispatch ---------------------------------------------
    def copy_dataset(self, name, src, dgrp, srcdir, keeps):
        if (self.repartition and src.name == self.plan.stack_vds_path
                and self._in_master(src)):
            # rebuild the image stack over the new data files (intercept
            # before the is_virtual / keeps branches below)
            self.write_repartition_vds(name, src, dgrp)
            return

        if is_virtual(src):
            self.copy_virtual(name, src, dgrp, srcdir)
            return

        entry = keeps.get(src.name)
        if entry is not None:
            if self.crop is not None:
                self._copy_stack_cropped(name, src, dgrp, entry.keep)
            else:
                self.copy_frames(name, src, dgrp, entry.keep, label="images")
            self.stats["frames"] += entry.keep
            return

        if self.is_dropped(name):
            if self.args.verbose:
                try:
                    size = human(src.nbytes)
                except Exception:
                    size = "?"
                print(f"  - dropping {src.name} {src.shape} ({size})")
            self.stats["dropped"] += 1
            return

        if self.is_mask(name, src):
            self.compress_dataset(name, src, dgrp)
            return

        if self.is_per_image(src):
            if self.args.verbose:
                print(f"  . trimming {src.name} "
                      f"{src.shape} -> ({self.nout}, ...)")
            self.copy_frames(name, src, dgrp, self.nout, label="per-image")
            self.stats["trimmed"] += 1
            return

        self.copy_plain(name, src, dgrp)

    def copy_virtual(self, name, src, dgrp, srcdir):
        img_shape = self._img_shape(src)         # cropped or original per-frame
        shape = (self.nout,) + img_shape
        layout = h5py.VirtualLayout(shape=shape, dtype=src.dtype)
        used = 0
        for vmap in src.virtual_sources():
            fname = vmap.file_name
            selfref = fname in (".", "")
            tgt = (self.master if selfref else (srcdir / fname).resolve())
            look_path, look_dpath = str(tgt), vmap.dset_name
            if selfref:
                # the plan keyed a self-ref-over-external-link source against
                # its real backing file; resolve the same hop to find it
                real_path, real_dpath = resolve_extlink(
                    src.file, vmap.dset_name, srcdir)
                if real_path is not None:
                    look_path, look_dpath = str(real_path), real_dpath
            entry = self.plan.sources.get((look_path, look_dpath))
            if entry is None or entry.keep == 0:
                continue
            vstart, vlen = span_of(vmap.vspace)
            sstart, _ = span_of(vmap.src_space)
            take = min(entry.keep - sstart, vlen, self.nout - vstart)
            if take <= 0:
                continue
            outfn = "." if selfref else self.enqueue(tgt)
            vsrc = h5py.VirtualSource(
                outfn, vmap.dset_name,
                shape=(entry.keep,) + img_shape, dtype=src.dtype)
            layout[vstart:vstart + take] = vsrc[sstart:sstart + take]
            used += take
        dst = dgrp.create_virtual_dataset(name, layout, fillvalue=src.fillvalue)
        copy_attrs(src, dst)
        self.fixup(name, src, dst, dgrp)
        if self.args.verbose:
            print(f"  . rebuilt VDS {src.name}: {src.shape} -> {shape} "
                  f"({used} frames mapped)")

    def _copy_chunks(self, src, dst, keep):
        """Move the leading `keep` planes chunk-by-chunk, compressed bytes
        untouched (`read_direct_chunk`/`write_direct_chunk`)."""
        missing = 0
        for offset in iter_chunk_offsets(src.shape, src.chunks, keep):
            try:
                mask, raw = src.id.read_direct_chunk(offset)
                dst.id.write_direct_chunk(offset, raw, mask)
            except OSError:
                missing += 1
        if missing and self.args.verbose:
            print(f"  . {src.name}: {missing} unallocated chunk(s) "
                  "left at fill value")

    def _create_verbatim(self, name, src, dgrp, shape):
        """Create `dst` with the source's *exact* ``cd_values`` so its
        compressed chunks can be copied byte-for-byte.

        This is the path for a filter whose ``set_local`` we cannot
        reproduce (unregistered in ``_preflight_pipelines``).  With the
        filter unregistered ``set_local`` does not run, so the recipe we
        pass is the recipe stored -- but HDF5 will not create a data set
        with a *mandatory* unregistered filter, so we flag the filter
        optional.  The compressed bytes are identical; the only difference
        from the source is that a reader lacking the filter would get raw
        bytes instead of an error (it could not use the data either way).
        Returns the data set, or None if the exact recipe did not come out
        (caller then falls back to decode + recompress).
        """
        want = filters_of(src.id.get_create_plist())
        opt = h5py.h5z.FLAG_OPTIONAL
        filters = tuple((c, f | opt, v) for c, f, v in want)
        try:
            dst = create_like(dgrp, name, src, shape, filters)
        except Exception:
            return None
        got = filters_of(dst.id.get_create_plist())
        # codes and cd_values must match exactly; only the optional bit differs
        ok = len(got) == len(want) and all(
            gc == wc and gv == wv and (gf & ~opt) == wf
            for (gc, gf, gv), (wc, wf, wv) in zip(got, want))
        if not ok:
            del dgrp[name]
            return None
        return dst

    def _create_image_like(self, dgrp, name, src, shape):
        """Create an image data set matching `src`'s dtype/chunks/pipeline.

        Returns ``(dst, mode)`` where mode is one of ``'raw'`` (exact pipeline
        reproduced -- copy chunks verbatim), ``'verbatim'`` (source cd_values
        stored, filter optional -- copy chunks verbatim), ``'recompress'``
        (gzip fallback -- caller must decode) or ``'plain'`` (uncompressed).
        Shared by ``copy_frames`` and the repartition writer so the create
        ladder -- and the item-1 pipeline assert -- lives in one place.
        """
        filters = self.pipelines.resolve(src) if src.chunks else None
        if src.chunks and filters is not None:
            # exact pipeline reproducible -- recreate it and copy chunks raw
            dst = create_like(dgrp, name, src, shape, filters)
            got = filters_of(dst.id.get_create_plist())
            want = filters_of(src.id.get_create_plist())
            if got != want:  # belt and braces -- must never happen
                raise SystemExit(
                    f"internal error: filter pipeline for {src.name} came out "
                    f"as {got}, expected {want}; refusing to write a data set "
                    "that would not decompress")
            return dst, "raw"
        if src.chunks:
            # exact recipe not reproducible (e.g. the local filter plugin is
            # a different version) -- still copy the compressed chunks
            # verbatim, storing the source's cd_values with the filter marked
            # optional.  No decode, no recompress, no plugin needed.
            dst = self._create_verbatim(name, src, dgrp, shape)
            if dst is not None:
                return dst, "verbatim"
            # last resort: decode + recompress (needs the filter plugin)
            print(f"  ! {src.name}: cannot reproduce filter pipeline "
                  "exactly and verbatim copy failed; falling back to "
                  "decode + recompress (needs the filter plugin, e.g. "
                  "pip install hdf5plugin)", file=sys.stderr)
            dst = dgrp.create_dataset(
                name, shape=shape, dtype=src.dtype, chunks=src.chunks,
                compression="gzip", compression_opts=1, shuffle=True)
            return dst, "recompress"
        return dgrp.create_dataset(name, shape=shape, dtype=src.dtype), "plain"

    def copy_frames(self, name, src, dgrp, keep, label=""):
        """Copy the leading `keep` planes of `src`, chunk-wise if possible."""
        keep = max(0, min(keep, src.shape[0] if src.ndim else 0))
        shape = (keep,) + tuple(src.shape[1:])
        dst, mode = self._create_image_like(dgrp, name, src, shape)
        if mode in ("raw", "verbatim"):
            if mode == "verbatim" and self.args.verbose:
                print(f"  . {src.name}: {keep} compressed frame(s) copied "
                      "verbatim (filter marked optional)")
            self._copy_chunks(src, dst, keep)
        elif mode == "recompress":
            for i in range(0, keep, max(1, src.chunks[0])):
                j = min(i + max(1, src.chunks[0]), keep)
                dst[i:j] = src[i:j]
        else:  # plain
            if keep:
                dst[...] = src[:keep]
        copy_attrs(src, dst)
        self.fixup(name, src, dst, dgrp)

    # -- crop (--crop) -------------------------------------------------
    def _copy_stack_cropped(self, name, src, dgrp, keep):
        """Copy the leading `keep` frames of an image stack, cropped to the
        central window.  Frames are decoded and recompressed (bitshuffle+LZ4,
        the native Eiger pipeline) -- a sub-window of a frame is not a whole
        chunk, so the verbatim copy path cannot be used here."""
        keep = max(0, min(keep, src.shape[0] if src.ndim else 0))
        shape = (keep,) + tuple(self.crop_shape)
        chunks = (1,) + tuple(self.crop_shape)
        dst = dgrp.create_dataset(name, shape=shape, dtype=src.dtype,
                                  chunks=chunks, **self._crop_compression())
        sy0, sy1, sx0, sx1 = self.crop
        for i in range(keep):
            dst[i] = src[i, sy0:sy1, sx0:sx1]
        copy_attrs(src, dst)
        self.fixup(name, src, dst, dgrp)
        if self.args.verbose:
            print(f"  . cropped stack {src.name} {tuple(src.shape)} -> {shape}")

    def _copy_detector_map_cropped(self, name, src, dgrp):
        """Copy a full-frame detector map (flatfield, correction table, ...),
        cropped to the central window."""
        try:
            data = self._crop2(src[...])
        except OSError as exc:
            print(f"  ! cannot read {src.name} ({exc}); copied uncropped",
                  file=sys.stderr)
            self.copy_plain(name, src, dgrp, _no_crop=True)
            return
        dst = dgrp.create_dataset(name, data=data, chunks=True,
                                  **self._crop_compression())
        copy_attrs(src, dst)
        self.fixup(name, src, dst, dgrp)
        if self.args.verbose:
            print(f"  . cropped map {src.name} {tuple(src.shape)} -> "
                  f"{data.shape}")

    def _crop_header_fixup(self, name, src, dst):
        """Adjust the geometry header for the crop: the image size shrinks and
        the detector origin (beam centre / module_offset) shifts by the number
        of pixels removed from the top-left corner."""
        sy0, _sy1, sx0, _sx1 = self.crop
        new_slow, new_fast = self.crop_shape
        try:
            if name == "beam_center_x":
                v = np.asarray(src[()], dtype=float) - sx0
                dst[...] = v
                if float(v.min()) < 0 or float(v.max()) > new_fast:
                    print(f"  ! beam_center_x {v.ravel()[0]:.1f} falls outside "
                          f"the cropped detector [0, {new_fast}]",
                          file=sys.stderr)
            elif name == "beam_center_y":
                v = np.asarray(src[()], dtype=float) - sy0
                dst[...] = v
                if float(v.min()) < 0 or float(v.max()) > new_slow:
                    print(f"  ! beam_center_y {v.ravel()[0]:.1f} falls outside "
                          f"the cropped detector [0, {new_slow}]",
                          file=sys.stderr)
            elif name == "x_pixels_in_detector":
                dst[()] = new_fast
            elif name == "y_pixels_in_detector":
                dst[()] = new_slow
            elif name == "data_size" and tuple(src.shape) == (2,):
                # NXmx detector_module/data_size is [slow, fast]
                dst[...] = np.array([new_slow, new_fast], dtype=dst.dtype)
            elif name == "module_offset":
                self._crop_module_offset(src, dst)
        except Exception as exc:
            print(f"  ! crop header fixup for {name} failed: {exc}",
                  file=sys.stderr)

    def _crop_module_offset(self, src, dst):
        """Shift the NXmx module_offset so the module origin tracks the new
        top-left pixel of the cropped frame.

        The module origin is at ``offset + magnitude * vector`` (a point in
        lab space); pixel (islow, ifast) sits at that origin plus
        ``ifast * fast_pixel_direction + islow * slow_pixel_direction``.  The
        cropped frame's (0, 0) pixel is the source's (sy0, sx0) pixel, so the
        new origin is the old origin displaced along the fast/slow pixel
        vectors by (sx0, sy0).  Re-decompose that point into magnitude+vector,
        keeping the (fixed) offset attribute unchanged.
        """
        grp = src.parent                          # detector/module group

        def to_m(x, units):
            if isinstance(units, bytes):
                units = units.decode("utf-8", "replace")
            return np.asarray(x, float) * {
                "m": 1.0, "mm": 1e-3, "um": 1e-6, "micron": 1e-6,
                "angstrom": 1e-10}.get(units, 1.0)

        def read_len(dname):
            d = grp[dname]
            mag = to_m(float(np.asarray(d[()]).ravel()[0]),
                       d.attrs.get("units", b"m"))
            vec = np.array(d.attrs["vector"], float)
            n = np.linalg.norm(vec)
            vec = vec / n if n else vec
            off = to_m(np.array(d.attrs.get("offset", [0.0, 0.0, 0.0]), float),
                       d.attrs.get("offset_units", b"m"))
            return mag, vec, off

        sy0, _sy1, sx0, _sx1 = self.crop
        fp_mag, fp_vec, _ = read_len("fast_pixel_direction")
        sp_mag, sp_vec, _ = read_len("slow_pixel_direction")
        mo_mag, mo_vec, mo_off = read_len("module_offset")
        origin = mo_off + mo_mag * mo_vec
        origin = origin + sx0 * fp_mag * fp_vec + sy0 * sp_mag * sp_vec
        rel = origin - mo_off
        new_mag_m = float(np.linalg.norm(rel))
        new_vec = rel / new_mag_m if new_mag_m else mo_vec
        # write the magnitude back in module_offset's own units
        units = src.attrs.get("units", b"m")
        if isinstance(units, bytes):
            units = units.decode("utf-8", "replace")
        factor = {"m": 1.0, "mm": 1e-3, "um": 1e-6}.get(units, 1.0)
        dst[()] = new_mag_m / factor
        old_vec = dst.attrs["vector"]
        dst.attrs.modify("vector", new_vec.astype(np.asarray(old_vec).dtype))

    # -- repartition (--frames-per-data) -------------------------------
    def _in_master(self, dset) -> bool:
        try:
            return Path(dset.file.filename).resolve() == self.master
        except Exception:
            return False

    def _copy_one_frame(self, sd, s_index, dd, d_index, mode):
        """Copy one image plane ``sd[s_index]`` -> ``dd[d_index]``."""
        if mode in ("raw", "verbatim"):
            # chunks[0] == 1 (enforced), so a frame is a whole number of
            # chunks; remap only the slow index and move the bytes raw
            base = (1,) + tuple(sd.shape[1:])
            for off in iter_chunk_offsets(base, sd.chunks, 1):
                s_off = (s_index,) + off[1:]
                d_off = (d_index,) + off[1:]
                try:
                    mask, raw = sd.id.read_direct_chunk(s_off)
                    dd.id.write_direct_chunk(d_off, raw, mask)
                except OSError:
                    pass                 # unallocated chunk -> leave fill value
        else:                            # recompress / plain -- decode + write
            dd[d_index] = sd[s_index]

    def _stamp_image_range(self, dst, lo):
        """If the template carried per-file image_nr_low/high, set them to the
        absolute scan range this block now covers (1-based)."""
        kept = dst.shape[0] if getattr(dst, "ndim", 0) else 0
        if "image_nr_low" in dst.attrs:
            try:
                dst.attrs.modify("image_nr_low", lo + 1)
            except Exception:
                pass
        if "image_nr_high" in dst.attrs:
            try:
                dst.attrs.modify("image_nr_high", lo + kept)
            except Exception:
                pass

    def _write_out_data_files(self):
        """Materialise the repartitioned image stack into new data files.

        Runs after ``_preflight_pipelines`` (so the verbatim path is armed)
        and before the master walk (so the rebuilt VDS references files that
        already exist).  Each output file holds up to ``--frames-per-data``
        frames, drawn -- possibly across a source boundary -- from the
        retained stack; compressed chunks are copied verbatim.
        """
        segs = self.plan.segments()
        # output-stack frame v  <-  (source, source frame index)
        frame_map = [None] * self.nout
        for s in segs:
            for i in range(s.rused):
                v = s.rvstart + i
                if 0 <= v < self.nout:
                    frame_map[v] = (s, s.sstart + i)

        handles: dict[str, h5py.File] = {}

        def sdset(s):
            fh = handles.get(str(s.path))
            if fh is None:
                fh = handles[str(s.path)] = h5py.File(s.path, "r")
            return fh[s.dset]

        # warn if an absorbed data file carries data sets other than the image
        # one (they are dropped when the file is repartitioned away)
        for s in segs:
            others: list[str] = []

            def _note(_n, obj, _keep=s.dset):
                if isinstance(obj, h5py.Dataset) and obj.name != _keep:
                    others.append(obj.name)

            try:
                sdset(s).file.visititems(_note)
            except Exception:
                pass                     # broken link etc. -- warning is advisory
            if others:
                print(f"  ! {s.path.name}: repartition keeps only {s.dset}; "
                      f"other data set(s) {others} are not carried over",
                      file=sys.stderr)

        dname = self.plan.data_internal_path.lstrip("/")   # 'data' or 'a/b/data'
        template = sdset(segs[0])
        try:
            for (lo, hi, fname, _lname) in self.repart_files:
                shape = (hi - lo,) + tuple(self.plan.image_shape)
                with h5py.File(self.outdir / fname, "w",
                               libver=self.libver) as fout:
                    if "/" in dname:
                        parent = fout.require_group(dname.rsplit("/", 1)[0])
                        leaf = dname.rsplit("/", 1)[1]
                    else:
                        parent, leaf = fout, dname
                    dst, mode = self._create_image_like(
                        parent, leaf, template, shape)
                    for v in range(lo, hi):
                        entry = frame_map[v]
                        if entry is None:        # gap -- leave at fill value
                            continue
                        s, si = entry
                        self._copy_one_frame(sdset(s), si, dst, v - lo, mode)
                    copy_attrs(template, dst)
                    self._stamp_image_range(dst, lo)
                self.stats["frames"] += (hi - lo)
                self.stats["files"] += 1
        finally:
            for fh in handles.values():
                fh.close()
        if self.args.verbose:
            print(f"  . repartitioned {self.nout} frame(s) into "
                  f"{len(self.repart_files)} file(s) of up to "
                  f"{self.repartition}")

    def write_repartition_vds(self, name, src, dgrp):
        """Rebuild the master's image VDS over the repartitioned data files.

        Preserves the DECTRIS structure: regenerate the ``data_00000i``
        external links in this NXdata group and a self-referencing VDS over
        them (source file ``'.'``), exactly as the source master carried --
        just with however many files the new partitioning needs.
        """
        img_shape = tuple(src.shape[1:])
        shape = (self.nout,) + img_shape
        dpath = self.plan.data_internal_path
        for (lo, hi, fname, lname) in self.repart_files:
            if lname in dgrp:                    # defensive: never double up
                del dgrp[lname]
            dgrp[lname] = h5py.ExternalLink(fname, dpath)
        layout = h5py.VirtualLayout(shape=shape, dtype=src.dtype)
        for (lo, hi, fname, lname) in self.repart_files:
            count = hi - lo                      # exact per-block length (item 3)
            vsrc = h5py.VirtualSource(
                ".", f"{self.plan.nxdata_group_path}/{lname}",
                shape=(count,) + img_shape, dtype=src.dtype)
            layout[lo:hi] = vsrc[0:count]
        dst = dgrp.create_virtual_dataset(name, layout, fillvalue=src.fillvalue)
        copy_attrs(src, dst)
        self.fixup(name, src, dst, dgrp)
        if self.args.verbose:
            print(f"  . rebuilt VDS {src.name}: {src.shape} -> {shape} "
                  f"across {len(self.repart_files)} data file(s)")

    def compress_dataset(self, name, src, dgrp):
        level = self.args.compression_level
        try:
            data = src[...]
        except OSError as exc:
            print(f"  ! cannot read {src.name} ({exc}); copying verbatim",
                  file=sys.stderr)
            self.copy_plain(name, src, dgrp)
            return
        if self.crop is not None and self._is_detector_2d(src):
            data = self._crop2(data)
            chunks = None                # source chunks may exceed the crop
        else:
            chunks = src.chunks
        if chunks is None:
            nbytes = data.dtype.itemsize
            for dim in data.shape:
                nbytes *= dim
            chunks = (tuple(data.shape) if nbytes <= 64 << 20 else True)
        dst = dgrp.create_dataset(name, data=data, chunks=chunks,
                                  shuffle=True, compression="gzip",
                                  compression_opts=level)
        copy_attrs(src, dst)
        self.stats["masks"] += 1
        if self.args.verbose:
            dgrp.file.flush()          # storage size is 0 until data is on disk
            before = src.id.get_storage_size() or data.nbytes
            after = dst.id.get_storage_size()
            ratio = f"{before / after:.0f}x" if after else "?"
            print(f"  . compressed {src.name} {human(before)} -> "
                  f"{human(after)} ({ratio})")

    def copy_plain(self, name, src, dgrp, _no_crop=False):
        if (self.crop is not None and not _no_crop
                and self._is_detector_2d(src) and src.ndim == 2):
            # a full-frame detector map that is neither an image stack nor a
            # recognised mask (e.g. a flatfield) -- crop it to the window
            self._copy_detector_map_cropped(name, src, dgrp)
            return
        if src.shape is None:            # NULL dataspace
            dst = dgrp.create_dataset(name, data=h5py.Empty(src.dtype))
            copy_attrs(src, dst)
            return
        if src.chunks:
            self.copy_frames(name, src, dgrp,
                             src.shape[0] if src.ndim else 0)
            return
        dst = dgrp.create_dataset(name, data=src[()], dtype=src.dtype,
                                  shape=src.shape)
        copy_attrs(src, dst)
        self.fixup(name, src, dst, dgrp)

    # -- book-keeping scalars ------------------------------------------
    def fixup(self, name, src, dst, dgrp):
        """Patch scalars that describe the length of the series."""
        if self.crop is not None and not self.args.no_fixup:
            self._crop_header_fixup(name, src, dst)
        if self.args.no_fixup or self.nout == self.nin:
            return
        try:
            if name == "image_nr_high" and src.size == 1:
                low = dgrp.get("image_nr_low")
                base = int(low[()]) if low is not None else 1
                dst[()] = base + self.nout - 1
            elif name in NIMAGE_SCALARS and src.size == 1:
                if int(src[()]) == self.nin:
                    dst[()] = self.nout
        except Exception:
            pass
        if "image_nr_high" in dst.attrs:
            # per-file attributes: the range this data set now covers
            kept = dst.shape[0] if getattr(dst, "ndim", 0) else self.nout
            try:
                low = int(dst.attrs.get("image_nr_low", 1))
                dst.attrs.modify("image_nr_high", low + kept - 1)
            except Exception:
                pass

    def verify(self):
        """Read the reduced stack back and compare it with the original."""
        out = self.outdir / self.outname_for(self.master)
        print("\nverifying ...")
        bad = 0
        # per-file image data: compare the trimmed copies with their sources
        for src in self.plan.sources.values():
            if src.keep == 0 or src.path == self.master:
                continue
            if src.path in self.absorbed_paths:
                # repartitioned: there is no 1:1 output file; the whole stack
                # is validated by the master-stack readback below
                continue
            dst = self.outdir / self.outname_for(src.path)
            try:
                with h5py.File(src.path, "r") as a, h5py.File(dst, "r") as b:
                    da, db = a[src.dset], b[src.dset]
                    same = (db.shape[0] == src.keep and
                            all(np.array_equal(self._src_frame(da, i), db[i])
                                for i in range(src.keep)))
                print(f"  {'ok  ' if same else 'FAIL'} {dst.name}:{src.dset}: "
                      f"{src.keep} frames")
                bad += 0 if same else 1
            except Exception as exc:
                print(f"  FAIL {dst.name}: {exc}")
                bad += 1
        try:
            with h5py.File(self.master, "r") as a, h5py.File(out, "r") as b:
                for path in sorted(self.plan.stack_paths):
                    da, db = a[path], b[path]
                    exp_frame = (tuple(self.crop_shape) if self.crop is not None
                                 else tuple(da.shape[1:]))
                    if db.shape != (self.nout,) + exp_frame:
                        print(f"  FAIL {path}: shape {db.shape}")
                        bad += 1
                        continue
                    for i in range(self.nout):
                        if not np.array_equal(self._src_frame(da, i), db[i]):
                            print(f"  FAIL {path}: frame {i} differs")
                            bad += 1
                            break
                    else:
                        print(f"  ok   {path}: {self.nout} frames identical")
        except Exception as exc:
            print(f"  FAIL could not read back: {exc}")
            return 1
        return 1 if bad else 0

    # -- summary --------------------------------------------------------
    def report(self):
        def tree_size(paths):
            return sum(p.stat().st_size for p in paths if p.exists())

        src_files = {Path(p) for p in self.done}
        src_files |= {s.path for s in self.plan.sources.values()}
        out_files = [self.outdir / n for n in self.outnames.values()]
        out_files += [self.outdir / f for (_, _, f, _) in self.repart_files]
        before, after = tree_size(src_files), tree_size(out_files)
        print(f"\nimages    : {self.nin} -> {self.nout}")
        if self.crop is not None:
            print(f"crop      : {tuple(self.plan.image_shape)} -> "
                  f"{tuple(self.crop_shape)} (slow, fast); {self.crop_info}")
        print(f"files     : {len(src_files)} in data set, "
              f"{self.stats['files']} written")
        print(f"masks     : {self.stats['masks']} recompressed")
        if self.stats["dropped"]:
            print(f"dropped   : {self.stats['dropped']} data set(s) removed")
        if self.repart_files:
            print(f"repart    : image stack -> {len(self.repart_files)} data "
                  f"file(s) of up to {self.repartition} frame(s)")
        print(f"per-image : {self.stats['trimmed']} data set(s) truncated")
        print(f"size      : {human(before)} -> {human(after)}"
              + (f"  ({before / after:.1f}x smaller)" if after else ""))
        print(f"output    : {self.outdir / self.outname_for(self.master)}")
        return 0


# --------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Reduce the size of an NXmx data set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The output is a complete, self-consistent NXmx data set: the "
               "master file plus trimmed copies of every file it links to.")
    p.add_argument("master", help="NXmx master file")
    p.add_argument("-n", "--num-images", type=int, required=True,
                   help="number of images to keep (from the start of the scan)")
    p.add_argument("-o", "--output", default="reduced",
                   help="output directory (default: ./reduced)")
    p.add_argument("--frames-per-data", type=int, default=None, metavar="N",
                   help="repartition the kept images into data files of at "
                        "most N frames each, rebuilding the virtual data set "
                        "(and the data_00000i links) to match; output files "
                        "continue the source's naming (e.g. ins10_1_00000i.h5). "
                        "Supports the DECTRIS self-referencing-VDS layout")
    p.add_argument("--crop", action="store_true",
                   help="crop the central area of an Eiger detector down to a "
                        "smaller class, adjusting the header (image size, beam "
                        "centre, module_offset).  Auto-selects the target "
                        "(16M->4M, 9M->1M); override with --crop-to.  Frames "
                        "are decoded and recompressed, so a filter plugin must "
                        "be installed; cannot be combined with --frames-per-data")
    p.add_argument("--crop-to", metavar="CLASS", default=None,
                   choices=tuple(EIGER_CLASS_GRID),
                   help="crop to this Eiger class instead of the default "
                        "(implies --crop); choices: "
                        + ", ".join(EIGER_CLASS_GRID))
    p.add_argument("-l", "--compression-level", type=int, default=4,
                   choices=range(0, 10),
                   help="gzip level for recompressed masks (default: 4)")
    p.add_argument("--also-compress", action="append", default=[],
                   metavar="GLOB",
                   help="additional data set names to gzip, e.g. 'flatfield' "
                        "(repeatable)")
    p.add_argument("--drop", action="append", default=[], metavar="GLOB",
                   help="data set names to omit from the output entirely, "
                        "e.g. 'flatfield' to drop the flat-field correction "
                        "(matched on the base name; image stacks are never "
                        "dropped; repeatable)")
    p.add_argument("--libver", default="v110",
                   choices=("v108", "v110", "v112", "v114", "latest"),
                   help="upper HDF5 format bound for the output (default: "
                        "v110 -- oldest that supports VDS; do not use 'latest' "
                        "unless every reader is as new as the writer)")
    p.add_argument("--verify", action="store_true",
                   help="after writing, read every retained frame back and "
                        "compare it with the original (needs any compression "
                        "filter plugin to be installed)")
    p.add_argument("--no-fixup", action="store_true",
                   help="do not patch nimages/image_nr_high book-keeping")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    args = p.parse_args(argv)

    if args.num_images < 1:
        p.error("-n must be at least 1")
    if args.frames_per_data is not None and args.frames_per_data < 1:
        p.error("--frames-per-data must be at least 1")
    if not Path(args.master).exists():
        p.error(f"no such file: {args.master}")

    return Reducer(args).run()


if __name__ == "__main__":
    sys.exit(main())
