import re
import string

try:
    from num2words import num2words
except ImportError:
    num2words = None


def expand_numbers(text):
    if num2words is None:
        return text

    def replace(match):
        number = match.group(0)
        try:
            if "." in number:
                return num2words(float(number))
            return num2words(int(number))
        except (ValueError, OverflowError):
            return number

    return re.sub(r"\d+(\.\d+)?", replace, text)


def strip_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_text(text, lowercase=False, remove_punctuation=False, expand_nums=False):
    result = text

    # expand numbers before stripping punctuation so decimals are read correctly
    if expand_nums:
        result = expand_numbers(result)

    if remove_punctuation:
        result = strip_punctuation(result)

    if lowercase:
        result = result.lower()

    result = re.sub(r"\s+", " ", result).strip()

    return result