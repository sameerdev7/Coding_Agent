def count_vowels(text: str) -> int:
    return sum(1 for char in text if char in "aeiou")


def reverse_words(sentence: str) -> str:
    return sentence[::-1]
