"""UI translation, Qt style: the source language is English, strings are marked in the code, locale files translate them.

In the code:
    from i18n import tr, tr_noop
    tr("Load")                                         # looked up by its English text
    tr("Cutting {i}/{n}: {name}", i=1, n=3, name=x)    # placeholders are filled AFTER translation
    tr_noop("text")                                    # only marks a string that is translated later by tr(variable)
A locale file is `locales/<lang>.json` = {"English text": "translation"}; a missing or empty translation shows the English text.
The language: `--lang` of ui.py, else the env variable UI_LANG, else the OS region / display language when a locale
file exists for it, else English.

Workflow (like lupdate / linguist), through the launcher (`run.bat i18n ...`) or `python i18n.py ...`:
    update            scan the code, add the new strings to every locales/*.json (empty = untranslated), list the obsolete ones
    update --prune    the same and drop the obsolete strings
    check             list untranslated, obsolete and placeholder-mismatching strings (exit code 1 when something is wrong)
    new <lang>        start a language file (e.g. `new de`), then fill in the empty values
Only calls with a string LITERAL as the first argument are found; tr(variable) is fine at run time but is not extracted,
so mark the literal with tr_noop where it is defined.
"""
import ast
import json
import os
import string
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LOCALES = os.path.join(ROOT, "locales")
SOURCES = ("ui.py", os.path.join("tools", "native_arrange.py"))          # files scanned for tr() / tr_noop()
MARKERS = {"tr", "tr_noop"}

_table = {}


def set_language(lang):
    """Load `locales/<lang>.json` (English or an unknown language = no translation)."""
    global _table
    _table = {}
    path = os.path.join(LOCALES, f"{lang}.json") if lang else ""
    if lang and lang != "en" and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _table = {k: v for k, v in json.load(f).items() if v}


def tr(text, /, **kw):
    """The translation of `text` (itself when there is none); `kw` fills the {placeholders} afterwards."""
    t = _table.get(text) or text
    return t.format(**kw) if kw else t


def tr_noop(text):
    """Marks a string for the extractor and returns it unchanged (translate it later with tr(variable))."""
    return text


def detect():
    """UI language: UI_LANG, else the first of (the OS region, the OS display language) that has a locale file, else English."""
    lang = os.environ.get("UI_LANG", "").strip()
    if lang:
        return lang
    candidates = []
    try:
        import ctypes
        import locale
        buf = ctypes.create_unicode_buffer(85)
        if ctypes.windll.kernel32.GetUserDefaultLocaleName(buf, len(buf)):
            candidates.append(buf.value)                                   # e.g. "uk-UA"
        candidates.append(locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), ""))
    except Exception:
        import locale
        candidates.append(locale.getlocale()[0] or "")
    have = set(locale_files())
    return next((c[:2].lower() for c in candidates if c[:2].lower() in have), "en")


def extract():
    """Every string literal passed to tr() / tr_noop() in the scanned files, in order of appearance, without repeats."""
    found = {}
    for rel in SOURCES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args
                 and (getattr(n.func, "id", None) in MARKERS or getattr(n.func, "attr", None) in MARKERS)]
        for n in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                found.setdefault(a.value, f"{rel}:{n.lineno}")
    return found


def locale_files():
    if not os.path.isdir(LOCALES):
        return []
    return sorted(f[:-5] for f in os.listdir(LOCALES) if f.endswith(".json"))


def read(lang):
    path = os.path.join(LOCALES, f"{lang}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write(lang, table):
    os.makedirs(LOCALES, exist_ok=True)
    with open(os.path.join(LOCALES, f"{lang}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
        f.write("\n")


def update(langs, prune=False):
    src = extract()
    for lang in langs:
        old = read(lang)
        new = {k: old.get(k, "") for k in src}                       # the order of the code
        obsolete = [k for k in old if k not in src]
        if not prune:
            new.update({k: old[k] for k in obsolete})
        write(lang, new)
        added = sum(1 for k in src if k not in old)
        print(f"{lang}: {len(src)} strings, {added} new, {sum(1 for v in new.values() if not v)} untranslated, "
              f"{len(obsolete)} obsolete{' (removed)' if prune else ''}")
        for k in obsolete:
            print(f"   obsolete: {k[:90]!r}")


def placeholders(text):
    return sorted(f for _, f, _, _ in string.Formatter().parse(text) if f)


def check(langs):
    src = extract()
    bad = 0
    for lang in langs:
        t = read(lang)
        missing = [k for k in src if k not in t]
        empty = [k for k in src if k in t and not t[k]]
        obsolete = [k for k in t if k not in src]
        broken = [k for k in src if t.get(k) and placeholders(k) != placeholders(t[k])]
        print(f"{lang}: {len(src) - len(missing) - len(empty)}/{len(src)} translated, {len(missing)} missing, "
              f"{len(empty)} empty, {len(obsolete)} obsolete, {len(broken)} with wrong {{placeholders}}")
        for k in missing + empty:
            print(f"   untranslated ({src[k]}): {k[:90]!r}")
        for k in obsolete:
            print(f"   obsolete: {k[:90]!r}")
        for k in broken:
            print(f"   placeholders differ: {k[:90]!r}")
        bad += len(missing) + len(empty) + len(broken)
    return bad


def main(argv):
    cmd = argv[0] if argv else "check"
    rest = argv[1:]
    prune = "--prune" in rest
    names = [a for a in rest if not a.startswith("--")]
    if cmd == "new":
        if len(names) != 1:
            sys.exit("Usage: i18n.py new <lang>")
        update(names)
    elif cmd == "update":
        update(names or locale_files(), prune)
    elif cmd == "check":
        sys.exit(1 if check(names or locale_files()) else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
