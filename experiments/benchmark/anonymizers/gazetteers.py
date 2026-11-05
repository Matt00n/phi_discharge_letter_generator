"""External name/place gazetteers for the rule-based baseline.

The lexica are drawn from the `Faker <https://faker.readthedocs.io>`_ package's
German locales (``de_DE``, ``de_AT``, ``de_CH``) rather than from any list
curated against our dataset. This keeps the baseline reproducible and, crucially,
free of *privileged tuning*: the entries were compiled from public general
knowledge and are entirely independent of the pools the generator draws from, so
the reported recall reflects a genuine out-of-the-box rule system rather than an
artefact of overlapping vocabularies.

Faker is imported lazily so the benchmark harness and config can be imported
without the experiment dependency group installed.
"""

from __future__ import annotations

from functools import lru_cache

#: Faker German locales combined for maximal, source-independent coverage.
_LOCALES = ("de_DE", "de_AT", "de_CH")


def _import_provider(family: str, locale: str):
    """Return the Faker provider class for ``family``/``locale`` (or ``None``)."""
    module = __import__(f"faker.providers.{family}.{locale}", fromlist=["Provider"])
    return getattr(module, "Provider", None)


@lru_cache(maxsize=1)
def person_gazetteers() -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(given_names, surnames)`` as casefolded frozensets.

    Given names merge the neutral, male, and female pools; hyphenated compounds
    (e.g. "Hans-Peter") are also split into their parts so a single component is
    enough to anchor a match.
    """
    given: set[str] = set()
    surnames: set[str] = set()
    for locale in _LOCALES:
        provider = _import_provider("person", locale)
        if provider is None:
            continue
        for attr in ("first_names", "first_names_male", "first_names_female"):
            for name in getattr(provider, attr, ()):
                _add_name_variants(name, given)
        for name in getattr(provider, "last_names", ()):
            _add_name_variants(name, surnames)
    return frozenset(given), frozenset(surnames)


@lru_cache(maxsize=1)
def city_gazetteer() -> frozenset[str]:
    """Return casefolded city names from the German Faker locales."""
    cities: set[str] = set()
    for locale in _LOCALES:
        provider = _import_provider("address", locale)
        if provider is None:
            continue
        for city in getattr(provider, "cities", ()):
            token = city.strip().casefold()
            if token:
                cities.add(token)
    return frozenset(cities)


@lru_cache(maxsize=1)
def street_suffixes() -> tuple[str, ...]:
    """Return distinct street suffixes (long and short) from the German locales.

    Sorted longest-first so a regex alternation prefers the more specific suffix
    (``straße`` before ``str.``).
    """
    suffixes: set[str] = set()
    for locale in _LOCALES:
        provider = _import_provider("address", locale)
        if provider is None:
            continue
        for attr in ("street_suffixes_long", "street_suffixes_short"):
            suffixes.update(s.strip() for s in getattr(provider, attr, ()) if s.strip())
    return tuple(sorted(suffixes, key=len, reverse=True))


def _add_name_variants(name: str, target: set[str]) -> None:
    """Add ``name`` (casefolded) and its hyphen-separated parts to ``target``."""
    token = name.strip()
    if not token:
        return
    target.add(token.casefold())
    if "-" in token:
        for part in token.split("-"):
            part = part.strip()
            if len(part) > 1:
                target.add(part.casefold())
