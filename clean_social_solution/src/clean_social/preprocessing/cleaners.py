from __future__ import annotations

import re
from importlib import resources

import nltk
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity
from textblob import TextBlob
from textblob.exceptions import MissingCorpusError

_URL_REGEX = re.compile(r"http\S+|www\S+", flags=re.IGNORECASE)
_PUNCT_REGEX = re.compile(r"[^\w\s]")
_WORD_REGEX = re.compile(r"[A-Za-z']+")
_TEXTBLOB_CORPORA_READY = False
_STOPWORDS_READY = False

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "none",
    "nobody",
    "nothing",
    "neither",
    "nor",
    "cannot",
    "cant",
    "can't",
    "don't",
    "dont",
    "didn't",
    "didnt",
    "isn't",
    "isnt",
    "wasn't",
    "wasnt",
    "won't",
    "wont",
}

# Covers common emoji ranges used in user-generated reviews.
_EMOJI_REGEX = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

try:
    import emoji
except Exception:  # pragma: no cover - fallback path
    emoji = None


def lowercase_text(text: str) -> str:
    return str(text).lower()


def remove_urls(text: str) -> str:
    return _URL_REGEX.sub("", str(text))


def remove_emojis(text: str) -> str:
    if emoji is not None:
        return emoji.replace_emoji(str(text), replace="")
    return _EMOJI_REGEX.sub("", str(text))


def remove_punctuation(text: str) -> str:
    return _PUNCT_REGEX.sub("", str(text))


def ensure_textblob_corpora() -> None:
    """Download required corpora once when TextBlob lemmatization is enabled."""
    global _TEXTBLOB_CORPORA_READY
    if _TEXTBLOB_CORPORA_READY:
        return

    # punkt_tab is required in newer NLTK versions used by TextBlob tokenization.
    for package in ("punkt_tab", "punkt", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)

    _TEXTBLOB_CORPORA_READY = True


def ensure_stopwords_ready() -> None:
    global _STOPWORDS_READY
    if _STOPWORDS_READY:
        return

    nltk.download("stopwords", quiet=True)
    _STOPWORDS_READY = True


def build_stopword_set(preserve_negations: bool = True) -> set[str]:
    ensure_stopwords_ready()
    words = set(stopwords.words("english"))
    if preserve_negations:
        words -= NEGATION_WORDS
    return words


def remove_stopwords_keep_negation(text: str, stopword_set: set[str] | None = None) -> str:
    source = str(text)
    words = stopword_set if stopword_set is not None else build_stopword_set(preserve_negations=True)
    cleaned_parts: list[str] = []
    last_end = 0

    for match in _WORD_REGEX.finditer(source):
        cleaned_parts.append(source[last_end:match.start()])
        token = match.group(0)
        if token.lower() not in words:
            cleaned_parts.append(token)
        last_end = match.end()

    cleaned_parts.append(source[last_end:])
    return "".join(cleaned_parts)


def lemmatize_textblob(text: str) -> str:
    try:
        blob = TextBlob(str(text))
        return " ".join(word.lemmatize() for word in blob.words)
    except MissingCorpusError:
        ensure_textblob_corpora()
        blob = TextBlob(str(text))
        return " ".join(word.lemmatize() for word in blob.words)


def build_symspell(max_dictionary_edit_distance: int = 2, prefix_length: int = 7) -> SymSpell:
    """Initialize SymSpell using the bundled English frequency dictionary."""
    symspell = SymSpell(max_dictionary_edit_distance=max_dictionary_edit_distance, prefix_length=prefix_length)

    dictionary_path = resources.files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
    loaded = symspell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
    if not loaded:
        raise RuntimeError("Failed to load SymSpell dictionary from symspellpy package.")
    return symspell


def correct_spelling_symspell(text: str, symspell: SymSpell, max_edit_distance: int = 2) -> str:
    """Correct word-level spelling while preserving non-word separators."""
    source = str(text)
    corrected_parts: list[str] = []
    last_end = 0

    for match in _WORD_REGEX.finditer(source):
        corrected_parts.append(source[last_end:match.start()])
        token = match.group(0)

        # Keep short tokens as-is to avoid over-correcting abbreviations.
        if len(token) <= 2:
            corrected_parts.append(token)
        else:
            suggestions = symspell.lookup(token.lower(), Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
            corrected_parts.append(suggestions[0].term if suggestions else token)

        last_end = match.end()

    corrected_parts.append(source[last_end:])
    return "".join(corrected_parts)
