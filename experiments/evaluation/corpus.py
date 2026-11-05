"""Schema-agnostic corpus ingestion with fail-fast validation."""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from experiments.evaluation.config import DatasetConfig, ProfileFieldConfig

ICD_PATTERN = re.compile(
    r"(?<![A-Z0-9])([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,2})?[!+*#]?)(?![A-Z0-9])",
    re.IGNORECASE,
)

# German OPS procedure codes, e.g. ``5-470`` or ``8-831.0`` — a single digit
# chapter, a hyphen, a two-to-three digit group, and an optional decimal tail.
OPS_PATTERN = re.compile(r"(?<![0-9])([0-9]-[0-9]{2,3}(?:\.[0-9a-z]{1,2})?)(?![0-9])")


@dataclass(frozen=True, slots=True)
class SpanSurface:
    """A single annotated mention: its surface string, label, and mention form."""

    surface: str
    label: str | None
    form: str | None


@dataclass(frozen=True, slots=True)
class Document:
    """The common representation consumed by all metrics."""

    doc_id: str
    text: str
    phi_span_count: int | None
    icd_text_codes: tuple[str, ...]
    icd_field_codes: tuple[str, ...]
    profile: tuple[str | float | tuple[str, ...] | None, ...] = ()
    phi_spans: tuple[SpanSurface, ...] = ()
    negative_spans: tuple[SpanSurface, ...] = ()
    section_sequence: tuple[str, ...] | None = None
    ops_text_codes: tuple[str, ...] = ()

    @property
    def icd_codes(self) -> tuple[str, ...]:
        """Union of codes observed in text and configured structured fields."""

        return tuple(sorted(set(self.icd_text_codes) | set(self.icd_field_codes)))


@dataclass(frozen=True, slots=True)
class Corpus:
    """A named collection and its loading metadata."""

    name: str
    documents: tuple[Document, ...]
    source_path: str
    language: str
    spacy_model: str | None
    icd_catalog_path: str | None
    profile_schema: tuple[ProfileFieldConfig, ...] = ()


def _open_text(path: Path, encoding: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding=encoding, newline="")
    return path.open("r", encoding=encoding, newline="")


def _nested_values(value: Any, path: str) -> list[Any]:
    """Resolve dotted keys; ``*`` traverses every list item or mapping value."""

    nodes = [value]
    if not path:
        return nodes
    for component in path.split("."):
        next_nodes: list[Any] = []
        for node in nodes:
            if component == "*":
                if isinstance(node, dict):
                    next_nodes.extend(node.values())
                elif isinstance(node, list | tuple):
                    next_nodes.extend(node)
            elif isinstance(node, dict) and component in node:
                next_nodes.append(node[component])
        nodes = next_nodes
    return nodes


def _one_value(record: dict[str, Any], path: str, *, required: bool) -> Any:
    values = _nested_values(record, path)
    if not values:
        if required:
            raise ValueError(f"required field {path!r} is missing")
        return None
    if len(values) != 1:
        raise ValueError(f"field {path!r} resolved to {len(values)} values")
    return values[0]


def _iter_jsonl(handle: TextIO, source: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"record at {source}:{line_number} is not an object")
        yield record


def _iter_records(config: DatasetConfig) -> Iterator[dict[str, Any]]:
    path = config.path
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist or is not a file: {path}")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-1:] == [".gz"] and len(suffixes) >= 2:
        effective_suffix = suffixes[-2]
    else:
        effective_suffix = suffixes[-1] if suffixes else ""
    file_format = config.format
    if file_format == "auto":
        file_format = {
            ".jsonl": "jsonl",
            ".ndjson": "jsonl",
            ".json": "json",
            ".csv": "csv",
            ".txt": "txt",
        }.get(effective_suffix, "")
        if not file_format:
            raise ValueError(f"cannot infer format from extension: {path}")

    with _open_text(path, config.encoding) as handle:
        if file_format == "jsonl":
            yield from _iter_jsonl(handle, path)
            return
        if file_format == "json":
            payload = json.load(handle)
            if config.records_field:
                values = _nested_values(payload, config.records_field)
                if len(values) != 1:
                    raise ValueError(
                        f"records_field {config.records_field!r} did not resolve once"
                    )
                payload = values[0]
            if not isinstance(payload, list):
                raise ValueError("JSON dataset must resolve to a list of objects")
            for index, record in enumerate(payload):
                if not isinstance(record, dict):
                    raise ValueError(f"JSON record {index} is not an object")
                yield record
            return
        if file_format == "csv":
            yield from csv.DictReader(handle)
            return
        if file_format == "txt":
            for index, line in enumerate(handle):
                yield {config.text_field: line.rstrip("\r\n"), "_line_id": index}
            return
    raise AssertionError(f"unhandled dataset format: {file_format}")


def canonical_icd(code: str) -> str | None:
    """Return a normalized ICD code when the entire value contains one."""

    match = ICD_PATTERN.search(code.strip().upper())
    if match is None:
        return None
    return match.group(1).upper()


def _flatten_scalars(values: Iterable[Any]) -> Iterator[str]:
    for value in values:
        if isinstance(value, dict):
            yield from _flatten_scalars(value.values())
        elif isinstance(value, list | tuple | set):
            yield from _flatten_scalars(value)
        elif value is not None:
            yield str(value)


def _canonical_profile_label(value: Any, *, casefold: bool) -> str:
    if not isinstance(value, str | int | float | bool):
        raise ValueError(
            "categorical and set profile values must resolve to scalar values"
        )
    label = " ".join(str(value).split())
    return label.casefold() if casefold else label


def _extract_profile_value(
    record: dict[str, Any], field: ProfileFieldConfig
) -> str | float | tuple[str, ...] | None:
    values = _nested_values(record, field.path)
    if not values:
        return None
    if field.kind == "set":
        labels = {
            _canonical_profile_label(value, casefold=field.casefold)
            for value in _flatten_scalars(values)
        }
        return tuple(sorted(labels))
    if len(values) != 1:
        raise ValueError(
            f"profile field {field.name!r} at {field.path!r} resolved to "
            f"{len(values)} values"
        )
    value = values[0]
    if value is None:
        return None
    if field.kind == "categorical":
        return _canonical_profile_label(value, casefold=field.casefold)
    if isinstance(value, bool):
        raise ValueError(f"numeric profile field {field.name!r} is boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"numeric profile field {field.name!r} is not numeric"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"numeric profile field {field.name!r} is not finite")
    if (
        field.numeric_min is not None
        and field.numeric_max is not None
        and not field.numeric_min <= number <= field.numeric_max
    ):
        raise ValueError(
            f"numeric profile field {field.name!r} value {number} lies outside "
            f"configured range [{field.numeric_min}, {field.numeric_max}]"
        )
    return number


def _extract_icd(
    record: dict[str, Any], text: str, config: DatasetConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    text_codes: set[str] = set()
    if config.extract_icd_from_text:
        text_codes.update(
            match.group(1).upper() for match in ICD_PATTERN.finditer(text)
        )
    field_codes: set[str] = set()
    for field in config.icd_fields:
        for scalar in _flatten_scalars(_nested_values(record, field)):
            field_codes.update(
                match.group(1).upper() for match in ICD_PATTERN.finditer(scalar)
            )
    return tuple(sorted(text_codes)), tuple(sorted(field_codes))


def _extract_ops(text: str, config: DatasetConfig) -> tuple[str, ...]:
    if not config.extract_ops_from_text:
        return ()
    return tuple(sorted({match.group(1) for match in OPS_PATTERN.finditer(text)}))


def _span_attribute(span: dict[str, Any], field_name: str | None) -> str | None:
    if field_name is None:
        return None
    value = _one_value(span, field_name, required=False)
    return None if value is None else str(value)


def _load_spans(
    record: dict[str, Any],
    text: str,
    config: DatasetConfig,
    *,
    spans_field: str | None,
    kind: str,
    required: bool,
    check_overlap: bool,
) -> tuple[int | None, tuple[SpanSurface, ...]]:
    """Validate offsets and lift each span's surface string, label, and form.

    Returns the annotation count (``None`` when ``spans_field`` is unset) and a
    tuple of :class:`SpanSurface`. Surface strings are only sliced when offset
    validation is enabled, because otherwise the offsets cannot be trusted; the
    label and mention form are captured regardless.
    """

    if spans_field is None:
        return None, ()
    spans = _one_value(record, spans_field, required=required)
    if spans is None:
        return 0, ()
    if not isinstance(spans, list):
        raise ValueError(f"field {spans_field!r} must be a list")

    surfaces: list[SpanSurface] = []
    offsets: list[tuple[int, int]] = []
    for span_index, span in enumerate(spans):
        if not isinstance(span, dict):
            raise ValueError(f"{kind} span {span_index} is not an object")
        label = _span_attribute(span, config.span_label_field)
        form = _span_attribute(span, config.span_form_field)
        if not config.validate_phi_offsets:
            surfaces.append(SpanSurface(surface="", label=label, form=form))
            continue
        start = _one_value(span, config.span_start_field, required=True)
        end = _one_value(span, config.span_end_field, required=True)
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError(f"{kind} span {span_index} start is not an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError(f"{kind} span {span_index} end is not an integer")
        if not 0 <= start < end <= len(text):
            raise ValueError(
                f"{kind} span {span_index} [{start}, {end}) is invalid for "
                f"{len(text)} characters"
            )
        offsets.append((start, end))
        surfaces.append(SpanSurface(surface=text[start:end], label=label, form=form))
    if check_overlap and config.validate_phi_offsets:
        ordered = sorted(offsets)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current[0] < previous[1]:
                raise ValueError(f"overlapping PHI spans: {previous} and {current}")
    return len(spans), tuple(surfaces)


def _section_sequence(
    record: dict[str, Any], config: DatasetConfig
) -> tuple[str, ...] | None:
    if config.section_sequence_field is None:
        return None
    value = _one_value(record, config.section_sequence_field, required=False)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            f"section_sequence_field {config.section_sequence_field!r} must be a list"
        )
    sections: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str | int | float | bool):
            raise ValueError(f"section {index} is not a scalar value")
        sections.append(" ".join(str(item).split()))
    return tuple(sections)


def load_corpus(config: DatasetConfig) -> Corpus:
    """Load one dataset and validate IDs, text, and optional PHI offsets."""

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(_iter_records(config)):
        raw_text = _one_value(record, config.text_field, required=True)
        if not isinstance(raw_text, str):
            raise ValueError(f"record {index} text is not a string")
        if config.id_field is None:
            doc_id = str(index)
        else:
            raw_id = _one_value(record, config.id_field, required=False)
            doc_id = str(index if raw_id is None else raw_id)
        if doc_id in seen_ids:
            raise ValueError(f"duplicate document ID in {config.name!r}: {doc_id!r}")
        seen_ids.add(doc_id)
        text_icd_codes, field_icd_codes = _extract_icd(record, raw_text, config)
        phi_count, phi_spans = _load_spans(
            record,
            raw_text,
            config,
            spans_field=config.phi_spans_field,
            kind="PHI",
            required=True,
            check_overlap=True,
        )
        _, negative_spans = _load_spans(
            record,
            raw_text,
            config,
            spans_field=config.negatives_field,
            kind="negative",
            required=False,
            check_overlap=False,
        )
        documents.append(
            Document(
                doc_id=doc_id,
                text=raw_text,
                phi_span_count=phi_count,
                icd_text_codes=text_icd_codes,
                icd_field_codes=field_icd_codes,
                profile=tuple(
                    _extract_profile_value(record, field)
                    for field in config.profile_fields
                ),
                phi_spans=phi_spans,
                negative_spans=negative_spans,
                section_sequence=_section_sequence(record, config),
                ops_text_codes=_extract_ops(raw_text, config),
            )
        )
    if not documents:
        raise ValueError(f"dataset {config.name!r} contains no documents")
    return Corpus(
        name=config.name,
        documents=tuple(documents),
        source_path=str(config.path),
        language=config.language,
        spacy_model=config.spacy_model,
        icd_catalog_path=(
            str(config.icd_catalog_path) if config.icd_catalog_path else None
        ),
        profile_schema=tuple(config.profile_fields),
    )
