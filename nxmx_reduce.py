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
"""

from __future__ import annotations

import argparse
import fnmatch
import os
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

__version__ = "1.1"

# Data set base names treated as masks (recompressed rather than copied).
DEFAULT_MASK_NAMES = ("pixel_mask", "mask", "*_mask")

# Scalar book-keeping data sets that describe the length of the series and
# so need patching when the series is truncated.
NIMAGE_SCALARS = ("nimages", "num_images", "number_of_images")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def human(nbytes: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes:.0f} B"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


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

    __slots__ = ("path", "dset", "length", "keep", "vstart", "sstart")

    def __init__(self, path: Path, dset: str, length: int):
        self.path = path
        self.dset = dset
        self.length = length
        self.keep = 0
        self.vstart = None      # offset in the virtual (stitched) stack
        self.sstart = 0         # first frame used from this source

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
            running = vstart + (src.length - sstart)

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
        self.libver = ("earliest", args.libver)
        self.pipelines = PipelineResolver(args.verbose)
        self.outnames: dict[str, str] = {}     # resolved src path -> out name
        self.queue: list[Path] = []
        self.done: set[str] = set()
        self.stats = {"masks": 0, "trimmed": 0, "frames": 0, "files": 0}
        # filters we unregistered so their chunks copy verbatim (see
        # _preflight_pipelines); re-registered before --verify
        self.verbatim_filters: set[int] = set()

    # -- naming --------------------------------------------------------
    def outname_for(self, path: Path) -> str:
        key = str(path)
        if key in self.outnames:
            return self.outnames[key]
        base = path.name
        stem, ext = os.path.splitext(base)
        n = 1
        while base in self.outnames.values():
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
        self._preflight_pipelines()
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
        if is_virtual(src):
            self.copy_virtual(name, src, dgrp, srcdir)
            return

        entry = keeps.get(src.name)
        if entry is not None:
            self.copy_frames(name, src, dgrp, entry.keep, label="images")
            self.stats["frames"] += entry.keep
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
        shape = (self.nout,) + tuple(src.shape[1:])
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
                shape=(entry.keep,) + tuple(src.shape[1:]), dtype=src.dtype)
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

    def copy_frames(self, name, src, dgrp, keep, label=""):
        """Copy the leading `keep` planes of `src`, chunk-wise if possible."""
        keep = max(0, min(keep, src.shape[0] if src.ndim else 0))
        shape = (keep,) + tuple(src.shape[1:])
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
            self._copy_chunks(src, dst, keep)
        elif src.chunks:
            # exact recipe not reproducible (e.g. the local filter plugin is
            # a different version) -- still copy the compressed chunks
            # verbatim, storing the source's cd_values with the filter marked
            # optional.  No decode, no recompress, no plugin needed.
            dst = self._create_verbatim(name, src, dgrp, shape)
            if dst is not None:
                if self.args.verbose:
                    print(f"  . {src.name}: {keep} compressed frame(s) copied "
                          "verbatim (filter marked optional)")
                self._copy_chunks(src, dst, keep)
            else:
                # last resort: decode + recompress (needs the filter plugin)
                print(f"  ! {src.name}: cannot reproduce filter pipeline "
                      "exactly and verbatim copy failed; falling back to "
                      "decode + recompress (needs the filter plugin, e.g. "
                      "pip install hdf5plugin)", file=sys.stderr)
                dst = dgrp.create_dataset(
                    name, shape=shape, dtype=src.dtype, chunks=src.chunks,
                    compression="gzip", compression_opts=1, shuffle=True)
                for i in range(0, keep, max(1, src.chunks[0])):
                    j = min(i + max(1, src.chunks[0]), keep)
                    dst[i:j] = src[i:j]
        else:
            dst = dgrp.create_dataset(name, shape=shape, dtype=src.dtype)
            if keep:
                dst[...] = src[:keep]
        copy_attrs(src, dst)
        self.fixup(name, src, dst, dgrp)

    def compress_dataset(self, name, src, dgrp):
        level = self.args.compression_level
        try:
            data = src[...]
        except OSError as exc:
            print(f"  ! cannot read {src.name} ({exc}); copying verbatim",
                  file=sys.stderr)
            self.copy_plain(name, src, dgrp)
            return
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

    def copy_plain(self, name, src, dgrp):
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
            dst = self.outdir / self.outname_for(src.path)
            try:
                with h5py.File(src.path, "r") as a, h5py.File(dst, "r") as b:
                    da, db = a[src.dset], b[src.dset]
                    same = (db.shape[0] == src.keep and
                            all(np.array_equal(da[i], db[i])
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
                    if db.shape != (self.nout,) + tuple(da.shape[1:]):
                        print(f"  FAIL {path}: shape {db.shape}")
                        bad += 1
                        continue
                    for i in range(self.nout):
                        if not np.array_equal(da[i], db[i]):
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
        before, after = tree_size(src_files), tree_size(out_files)
        print(f"\nimages    : {self.nin} -> {self.nout}")
        print(f"files     : {len(src_files)} in data set, "
              f"{self.stats['files']} written")
        print(f"masks     : {self.stats['masks']} recompressed")
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
    p.add_argument("-l", "--compression-level", type=int, default=4,
                   choices=range(0, 10),
                   help="gzip level for recompressed masks (default: 4)")
    p.add_argument("--also-compress", action="append", default=[],
                   metavar="GLOB",
                   help="additional data set names to gzip, e.g. 'flatfield' "
                        "(repeatable)")
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
    if not Path(args.master).exists():
        p.error(f"no such file: {args.master}")

    return Reducer(args).run()


if __name__ == "__main__":
    sys.exit(main())
