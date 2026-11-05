"""Format-varied samplers for atomic (non-person) PII values.

Each builder returns a ``variants`` dict ready for :class:`render.AtomicEntity`.
Surface-form variation within formal constraints (date formats, spaced vs.
unspaced identifiers, German vs. foreign phone formats) is a deliberate
diversity axis: an anonymizer must generalize across formats, not memorize one.
"""

from __future__ import annotations

import random
from datetime import date

_MONTHS_DE = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]


def date_variants(d: date) -> dict[str, str]:
    """Return multiple valid German surface forms for a date."""
    return {
        "default": d.strftime("%d.%m.%Y"),  # 14.05.1963
        "long": d.strftime("%d.%m.%Y"),
        "short": f"{d.day}.{d.month}.{d.strftime('%y')}",  # 14.5.63
        "iso": d.strftime("%Y-%m-%d"),  # 1963-05-14
        "textual": f"{d.day}. {_MONTHS_DE[d.month - 1]} {d.year}",  # 14. Mai 1963
        "spaced": d.strftime("%d. %m. %Y"),
    }


def kv_number(rng: random.Random) -> dict[str, str]:
    """German Versichertennummer: 1 uppercase letter + 9 digits."""
    letter = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(str(rng.randint(0, 9)) for _ in range(9))
    plain = f"{letter}{digits}"
    return {
        "default": plain,
        "spaced": f"{letter} {digits[:4]} {digits[4:]}",
        "dashed": f"{letter}-{digits[:3]}-{digits[3:6]}-{digits[6:]}",
        "prefixed": f"KV-Nr. {plain}",
        "compact": plain,
    }


def case_number(rng: random.Random) -> dict[str, str]:
    n = rng.randint(1000000, 9999999)
    return {
        "default": f"{n}",
        "prefixed": f"Fall-Nr. {n}",
        "fid": f"FID-{n}",
    }


def patient_id(rng: random.Random) -> dict[str, str]:
    n = rng.randint(100000, 999999)
    return {
        "default": f"{n}",
        "prefixed": f"Pat.-ID {n}",
        "pid": f"PID-{n}",
    }


def phone(rng: random.Random, area_code: str | None = None) -> dict[str, str]:
    # Prefer the caller-supplied (city-consistent) area code; otherwise fall back
    # to a random one (used when the location has no known code, e.g. tricky towns).
    area = area_code or rng.choice(
        [
            "030",
            "089",
            "040",
            "0221",
            "069",
            "0711",
            "0341",
            "0351",
            "0201",
            "0231",
            "0511",
            "0421",
            "0621",
            "0911",
            "0241",
            "0361",
        ]
    )
    rest = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(6, 8)))
    return {
        "default": f"{area} {rest}",
        "slashed": f"{area} / {rest}",
        "intl": f"+49 {area.lstrip('0')} {rest}",
        "dashed": f"{area}-{rest}",
        "compact": f"{area}{rest}",
        "parens": f"({area}) {rest}",
    }


def mobile(rng: random.Random) -> dict[str, str]:
    prefix = rng.choice(
        ["0151", "0152", "0157", "0160", "0170", "0171", "0176", "0175", "0162", "0163"]
    )
    rest = "".join(str(rng.randint(0, 9)) for _ in range(7))
    return {
        "default": f"{prefix} {rest}",
        "intl": f"+49 {prefix.lstrip('0')} {rest}",
        "dashed": f"{prefix}-{rest[:3]}-{rest[3:]}",
        "compact": f"{prefix}{rest}",
        "grouped": f"{prefix} {rest[:3]} {rest[3:]}",
    }


def fax(rng: random.Random, area_code: str | None = None) -> dict[str, str]:
    """Fax number — same structure as phone but typically different formatting."""
    area = area_code or rng.choice(
        ["030", "089", "040", "0221", "069", "0711", "0341", "0351"]
    )
    rest = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(6, 8)))
    return {
        "default": f"{area} {rest}",
        "slashed": f"{area} / {rest}",
        "intl": f"+49 {area.lstrip('0')} {rest}",
        "dashed": f"{area}-{rest}",
        "prefixed": f"Fax {area} {rest}",
        "compact": f"{area}{rest}",
    }


def url(rng: random.Random, institution: str) -> dict[str, str]:
    """Hospital / practice website URL with varied formats."""
    # Normalize institution to a plausible domain slug
    slug = institution.lower()
    for ch in "äöüß":
        slug = slug.replace(ch, {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}[ch])
    slug = "".join(c if c.isalnum() else "-" for c in slug).strip("-")
    # Take first 2-3 meaningful words
    parts = [p for p in slug.split("-") if len(p) > 2][:3]
    domain = "-".join(parts) if parts else "klinikum"
    tld = rng.choice([".de", ".de", ".de", ".com", ".org"])
    base = f"{domain}{tld}"
    return {
        "default": f"www.{base}",
        "https": f"https://www.{base}",
        "bare": base,
        "path": f"www.{base}/{rng.choice(['kontakt', 'team', 'kliniken', 'ueber-uns'])}",
    }


def lanr(rng: random.Random) -> dict[str, str]:
    """Lebenslange Arztnummer (LANR): 9 digits, first 7 = personal, last 2 = site."""
    digits = "".join(str(rng.randint(0, 9)) for _ in range(9))
    return {
        "default": digits,
        "spaced": f"{digits[:3]} {digits[3:6]} {digits[6:]}",
        "grouped": f"{digits[:7]}-{digits[7:]}",
        "prefixed": f"LANR {digits}",
        "labeled": f"Arztnr. {digits}",
    }


def bsnr(rng: random.Random) -> dict[str, str]:
    """Betriebsstättennummer (BSNR): 9 digits."""
    digits = "".join(str(rng.randint(0, 9)) for _ in range(9))
    return {
        "default": digits,
        "spaced": f"{digits[:2]} {digits[2:5]} {digits[5:]}",
        "prefixed": f"BSNR {digits}",
        "labeled": f"Betriebsstätten-Nr. {digits}",
    }


def ik_nummer(rng: random.Random) -> dict[str, str]:
    """Institutionskennzeichen (IK): 9 digits, starts with institution type."""
    # First 2 digits encode type (10=Krankenhaus, 26=Arztpraxis, etc.)
    prefix = rng.choice(["10", "26", "30", "40", "66"])
    rest = "".join(str(rng.randint(0, 9)) for _ in range(7))
    digits = prefix + rest
    return {
        "default": digits,
        "spaced": f"{digits[:3]} {digits[3:6]} {digits[6:]}",
        "prefixed": f"IK {digits}",
        "labeled": f"IK-Nr. {digits}",
    }


def email(rng: random.Random, vorname: str, nachname: str) -> dict[str, str]:
    def norm(s: str) -> str:
        table = str.maketrans("äöüßÄÖÜ", "aousAOU")
        return s.lower().translate(table).replace(" ", "").replace("-", "")

    v, n = norm(vorname.split()[0]), norm(nachname.split()[-1])
    domain = rng.choice(["web.de", "gmx.de", "gmail.com", "t-online.de", "outlook.de"])
    return {
        "default": f"{v}.{n}@{domain}",
        "initial": f"{v[0]}.{n}@{domain}",
        "underscore": f"{v}_{n}@{domain}",
    }


# Streets and city/PLZ pools (extend freely).
STREETS = [
    # -straße (most common German suffix)
    "Hauptstraße",
    "Bahnhofstraße",
    "Schillerstraße",
    "Goethestraße",
    "Gartenstraße",
    "Ringstraße",
    "Rheinstraße",
    "Friedrichstraße",
    "Berliner Straße",
    "Beethovenstraße",
    "Mozartstraße",
    "Kirchstraße",
    "Schulstraße",
    "Dorfstraße",
    "Industriestraße",
    "Blumenstraße",
    "Bergstraße",
    "Waldstraße",
    "Bismarckstraße",
    "Kaiserstraße",
    "Lessingstraße",
    "Herderstraße",
    "Parkstraße",
    "Schlossstraße",
    "Rathausstraße",
    "Jahnstraße",
    "Steinstraße",
    "Brückenstraße",
    "Neue Straße",
    "Wiesenstraße",
    # -weg
    "Birkenweg",
    "Erlenweg",
    "Fichtenweg",
    "Tannenweg",
    "Buchenweg",
    "Eichenweg",
    "Lindenweg",
    "Ahornweg",
    "Kastanienweg",
    "Rosenweg",
    "Feldweg",
    "Waldweg",
    "Höhenweg",
    "Wiesenweg",
    "Uferweg",
    # -allee
    "Lindenallee",
    "Kastanienallee",
    "Pappelallee",
    "Schloßallee",
    "Königsallee",
    "Sophienallee",
    "Friedensallee",
    # -gasse
    "Marktgasse",
    "Kirchgasse",
    "Badgasse",
    "Klostergasse",
    "Herrengasse",
    "Judengasse",
    "Schulgasse",
    # -ring / -damm / -platz
    "Hohenzollernring",
    "Sachsenring",
    "Hansaring",
    "Theodor-Heuss-Ring",
    "Friedrich-Ebert-Damm",
    "Elbdamm",
    "Hammer Deich",
    "Marktplatz",
    "Rathausplatz",
    "Domplatz",
    "Schillerplatz",
    "Karlsplatz",
    # Person-named streets
    "Adenauer-Allee",
    "Konrad-Adenauer-Straße",
    "Willy-Brandt-Platz",
    "Heinrich-Heine-Allee",
    "Thomas-Mann-Straße",
    "Albert-Einstein-Straße",
    "Robert-Koch-Straße",
    "Max-Planck-Straße",
    "Clara-Zetkin-Straße",
    "Rosa-Luxemburg-Straße",
    "Kurt-Schumacher-Straße",
    # Landmark / geographic
    "Am Stadtpark",
    "Am Markt",
    "Am Hafen",
    "Am Mühlenberg",
    "Am Sonnenhang",
    "Zur Mühle",
    "An der Kirche",
    "Am Alten Rathaus",
    "Im Brühl",
    "Auf dem Kamp",
    "Hinter der Mauer",
    # Boulevard / -ufer / -chaussee
    "Kurfürstendamm",
    "Leopoldstraße",
    "Maximilianstraße",
    "Alsterufer",
    "Elbchaussee",
    "Rheinuferstraße",
]

# (city, PLZ pool, telephone area code). The area code keeps letterhead / contact
# phone and fax numbers geographically consistent with the city (see ``phone`` /
# ``fax`` / ``address``).
CITY_PLZ = [
    # All 16 Bundesländer represented
    # BW
    ("Stuttgart", ["70173", "70176", "70190", "70469"], "0711"),
    ("Mannheim", ["68159", "68161", "68165", "68167"], "0621"),
    ("Freiburg", ["79098", "79100", "79104", "79106"], "0761"),
    ("Heidelberg", ["69115", "69117", "69120", "69126"], "06221"),
    ("Karlsruhe", ["76131", "76133", "76135", "76137"], "0721"),
    # BY
    ("München", ["80331", "81675", "80802", "81373"], "089"),
    ("Nürnberg", ["90402", "90403", "90439", "90489"], "0911"),
    ("Augsburg", ["86150", "86152", "86153", "86156"], "0821"),
    ("Regensburg", ["93047", "93049", "93051", "93053"], "0941"),
    # BE
    ("Berlin", ["10115", "10785", "13353", "12043", "10969", "12163"], "030"),
    # BB
    ("Potsdam", ["14467", "14469", "14471", "14473"], "0331"),
    ("Cottbus", ["03042", "03044", "03046", "03050"], "0355"),
    # HB
    ("Bremen", ["28195", "28199", "28203", "28209", "28357"], "0421"),
    # HH
    ("Hamburg", ["20095", "22303", "20354", "21073", "22041"], "040"),
    # HE
    ("Frankfurt", ["60311", "60313", "60316", "60318", "60594"], "069"),
    ("Wiesbaden", ["65183", "65185", "65187", "65189"], "0611"),
    ("Kassel", ["34117", "34119", "34121", "34125"], "0561"),
    # MV
    ("Rostock", ["18055", "18057", "18059", "18069"], "0381"),
    ("Schwerin", ["19053", "19055", "19057", "19059"], "0385"),
    # NI
    ("Hannover", ["30159", "30161", "30163", "30167", "30449"], "0511"),
    ("Braunschweig", ["38100", "38102", "38104", "38106"], "0531"),
    ("Göttingen", ["37073", "37075", "37077", "37079"], "0551"),
    ("Oldenburg", ["26121", "26122", "26125", "26127"], "0441"),
    # NW
    ("Köln", ["50667", "50668", "50670", "50823", "50937"], "0221"),
    ("Düsseldorf", ["40210", "40211", "40213", "40215", "40217"], "0211"),
    ("Dortmund", ["44135", "44137", "44139", "44229", "44339"], "0231"),
    ("Essen", ["45127", "45128", "45130", "45131"], "0201"),
    ("Bonn", ["53111", "53113", "53115", "53117"], "0228"),
    ("Münster", ["48143", "48145", "48147", "48149"], "0251"),
    ("Aachen", ["52062", "52064", "52066", "52068"], "0241"),
    ("Bielefeld", ["33602", "33604", "33607", "33609"], "0521"),
    # RP
    ("Mainz", ["55116", "55118", "55120", "55122"], "06131"),
    ("Koblenz", ["56068", "56070", "56072", "56073"], "0261"),
    ("Trier", ["54290", "54292", "54293", "54294"], "0651"),
    # SL
    ("Saarbrücken", ["66111", "66113", "66115", "66117"], "0681"),
    # SN
    ("Leipzig", ["04109", "04275", "04277", "04317", "04229"], "0341"),
    ("Dresden", ["01067", "01069", "01097", "01099", "01187"], "0351"),
    ("Chemnitz", ["09111", "09112", "09113", "09116"], "0371"),
    # ST
    ("Halle", ["06108", "06110", "06112", "06114"], "0345"),
    ("Magdeburg", ["39104", "39106", "39108", "39110"], "0391"),
    # SH
    ("Kiel", ["24103", "24105", "24107", "24109"], "0431"),
    ("Lübeck", ["23552", "23554", "23556", "23558"], "0451"),
    ("Flensburg", ["24937", "24939", "24941", "24943"], "0461"),
    # TH
    ("Erfurt", ["99084", "99086", "99089", "99091"], "0361"),
    ("Jena", ["07743", "07745", "07747", "07749"], "03641"),
    ("Weimar", ["99423", "99425", "99427", "99428"], "03643"),
]

# Tricky / unusual village and town names that look like person names, common
# words, or are otherwise confusing for NER / anonymizers. Used only in hard cases.
TRICKY_LOCATIONS: list[tuple[str, str]] = [
    # Look like surnames
    ("Adenau", "53518"),
    ("Bismark", "39629"),
    ("Moser", "83109"),
    ("Hauser", "83254"),
    ("Bauer", "94501"),
    ("Kirchner", "07907"),
    ("Hagen", "58089"),
    ("Berg", "51702"),
    ("Horn", "21514"),
    ("Worms", "67547"),
    ("Stein", "24235"),
    ("Nagel", "95697"),
    ("Bach", "84172"),
    ("Kahl", "63796"),
    ("Ernst", "56340"),
    ("Hahn", "56357"),
    ("Baar", "78056"),
    # Look like first names
    ("Annaberg-Buchholz", "09456"),
    ("Marienberg", "09496"),
    ("Friedrichshafen", "88045"),
    ("Ludwigshafen", "67059"),
    ("Karlstadt", "97753"),
    ("Petersberg", "36100"),
    ("Martinfeld", "37318"),
    ("Ottweiler", "66564"),
    # Look like common words (confusing)
    ("Lachen", "67435"),
    ("Elend", "38875"),
    ("Sorge", "38875"),
    ("Pech", "54668"),
    ("Wald", "93192"),
    ("Wasser", "55234"),
    ("Leer", "26789"),
    ("Furth", "84095"),
    ("Au", "79280"),
    ("Oy-Mittelberg", "87466"),
    ("Grub", "83257"),
    ("Blender", "27337"),
    ("Trier-Land", "54296"),
    # Very short / odd
    ("Aub", "97239"),
    ("Aue", "08280"),
    ("Auma", "07955"),
    ("Oy", "87466"),
    ("Au am Rhein", "76474"),
    # Compound names that look institution-shaped
    ("Bad Homburg", "61348"),
    ("Bad Kreuznach", "55543"),
    ("Bad Kissingen", "97688"),
    ("Neustadt an der Weinstraße", "67433"),
    ("Markt Schwaben", "85570"),
    ("Kirchheim unter Teck", "73230"),
]


def address(
    rng: random.Random,
    *,
    difficulty: str = "medium",
    tricky_prob: float = 0.35,
    house_number_range: tuple[int, int] | list[int] = (1, 180),
    house_suffix_prob: float = 0.2,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str | None]:
    """Return (street_address, zip, city, area_code) that are mutually consistent.

    ``area_code`` is the city's telephone area code (or ``None`` for tricky
    locations, which have no known code and intentionally stay unconstrained).
    On hard difficulty, ~35% chance of picking from the tricky locations pool.
    """
    use_tricky = difficulty == "hard" and rng.random() < tricky_prob
    if use_tricky:
        city, plz = rng.choice(TRICKY_LOCATIONS)
        area_code = None
    else:
        city, plz_pool, area_code = rng.choice(CITY_PLZ)
        plz = rng.choice(plz_pool)
    street = rng.choice(STREETS)
    num = str(rng.randint(*house_number_range))
    if rng.random() < house_suffix_prob:
        num += rng.choice(["a", "b", "c"])

    street_addr = {
        "default": f"{street} {num}",
        "abbrev": f"{street.replace('straße', 'str.')} {num}",
    }
    zip_v = {"default": plz}
    city_v = {"default": city}
    return street_addr, zip_v, city_v, area_code


# --- Non-PII identifier samplers (over-redaction traps for numbers) -----------


def pzn(rng: random.Random) -> str:
    """Pharmazentralnummer: 8 digits (PZN-xxxxxxxx)."""
    digits = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return rng.choice([f"PZN {digits}", f"PZN-{digits}", digits])


def atc_code(rng: random.Random) -> str:
    """ATC code: 1 letter + 2 digits + 2 letters + 2 digits (e.g. N02BE01)."""
    l1 = rng.choice("ABCDGHJLMNPRSVX")
    d1 = f"{rng.randint(1, 9):02d}"
    l2 = rng.choice("ABCDEFGHX") + rng.choice("ABCDEFGHX")
    d2 = f"{rng.randint(1, 15):02d}"
    return f"{l1}{d1}{l2}{d2}"


def drg_code(rng: random.Random) -> str:
    """DRG code: 1 letter + 2 digits + 1 letter (e.g. F62B, I68A)."""
    l1 = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    d = f"{rng.randint(1, 99):02d}"
    l2 = rng.choice("ABCDEFGHIJZ")
    return f"{l1}{d}{l2}"


def ean_code(rng: random.Random) -> str:
    """EAN/GTIN barcode (13 digits) — appears on medication packaging."""
    digits = "".join(str(rng.randint(0, 9)) for _ in range(13))
    return digits


def charge_number(rng: random.Random) -> str:
    """Medication lot/batch number (Chargen-Nr.): alphanumeric, 6-10 chars."""
    length = rng.randint(6, 10)
    chars = "".join(
        rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(length)
    )
    return rng.choice([f"Ch.-B.: {chars}", f"Charge {chars}", chars])


def ops_code_surface(rng: random.Random) -> str:
    """OPS procedure code (e.g. 5-820.01). Already in clinical text, not PII."""
    chapter = rng.choice(["1", "3", "5", "6", "8", "9"])
    group = f"{rng.randint(1, 99):02d}"
    sub = rng.choice(
        ["", f".{rng.randint(0, 9)}", f".{rng.randint(0, 9)}{rng.randint(0, 9)}"]
    )
    return f"{chapter}-{group}{sub}"


def icd_code_surface(rng: random.Random) -> str:
    """ICD-10-GM code (e.g. I25.1, E11.9). Medical, not PII."""
    letter = rng.choice("ABCDEFGHIJKLMNOPQRSTUVZ")
    num = f"{rng.randint(0, 99):02d}"
    sub = rng.choice(["", f".{rng.randint(0, 9)}"])
    return f"{letter}{num}{sub}"


# fmt: off
# Pool of non-PII IDs with their contextual label (how they appear in text)
NONPII_ID_TYPES: list[tuple[str, callable]] = [
    ("PZN (Pharmazentralnummer)", pzn),
    ("ATC-Code", atc_code),
    ("DRG-Code", drg_code),
    ("EAN/GTIN", ean_code),
    ("Chargen-Nr.", charge_number),
    ("OPS-Code", ops_code_surface),
    ("ICD-10-Code", icd_code_surface),
]
# fmt: on


def sample_nonpii_ids(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """Sample n non-PII IDs. Returns list of (type_label, surface_string)."""
    n = max(0, min(n, len(NONPII_ID_TYPES) * 2))
    results: list[tuple[str, str]] = []
    for _ in range(n):
        type_label, generator = rng.choice(NONPII_ID_TYPES)
        results.append((type_label, generator(rng)))
    return results


def sample_colliding_nonpii_id(
    rng: random.Random,
    pii_surface: str,
    mode_weights: list[float] | None = None,
) -> tuple[str, str]:
    """Generate a non-PII ID whose digit pattern overlaps a PII number.

    Used in adversarial mode. Three collision strengths are sampled:
    * **exact** (40%) — the non-PII surface contains the identical digit string
      as the PII value, wrapped in a non-PII label (e.g. PII ``"4567890"`` →
      ``"PZN 4567890"``). Maximum difficulty for digit-matching heuristics.
    * **substring** (35%) — a long digit run from the PII value is embedded
      inside a longer non-PII number.
    * **padded** (25%) — a digit run is padded/truncated into a fixed-width
      non-PII format.
    """
    import re as _re

    # Extract digit sequences from PII surface (≥3 digits)
    digit_runs = [m.group() for m in _re.finditer(r"\d{3,}", pii_surface)]
    if not digit_runs:
        type_label, generator = rng.choice(NONPII_ID_TYPES)
        return type_label, generator(rng)

    shared_digits = rng.choice(digit_runs)

    # --- exact: same digit string, different non-PII context ---------------
    # fmt: off
    _exact_wrappers = [
        lambda d: f"PZN {d}",
        lambda d: f"PZN-{d}",
        lambda d: f"Ch.-B.: {d}",
        lambda d: f"Chargen-Nr. {d}",
        lambda d: f"Ref.-Nr. {d}",
        lambda d: f"Bestell-Nr. {d}",
        lambda d: f"Rezept-Nr. {d}",
        lambda d: f"Abrechnungsziffer {d}",
    ]
    # fmt: on

    # --- substring: digit run embedded in a longer number ------------------
    def _substring_wrap(d: str) -> str:
        pre = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(1, 3)))
        suf = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(1, 3)))
        full = pre + d + suf
        return rng.choice([f"EAN {full}", f"PZN {full}", f"Serien-Nr. {full}"])

    # --- padded: digit run truncated/padded into fixed format --------------
    def _padded_wrap(d: str) -> str:
        return rng.choice(
            [
                f"PZN {d[:8].ljust(8, str(rng.randint(0, 9)))}",
                f"Ch.-B.: {rng.choice('ABCDEFGH')}{d[:6]}{rng.choice('ABCDEFGH')}",
                f"{d[:7].ljust(7, str(rng.randint(0, 9)))}-{rng.randint(10, 99)}",
                f"EAN {d[:8].ljust(8, str(rng.randint(0, 9)))}{rng.randint(10000, 99999)}",
            ]
        )

    mode = rng.choices(
        ["exact", "substring", "padded"],
        weights=mode_weights if mode_weights is not None else [0.40, 0.35, 0.25],
        k=1,
    )[0]

    if mode == "exact":
        wrapper = rng.choice(_exact_wrappers)
        surface = wrapper(shared_digits)
    elif mode == "substring":
        surface = _substring_wrap(shared_digits)
    else:
        surface = _padded_wrap(shared_digits)

    return "Kollidierende Nr.", surface
