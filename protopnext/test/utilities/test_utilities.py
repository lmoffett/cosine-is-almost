import random
from typing import List

from protopnet.utilities.naming import ADJECTIVES, NOUNS, generate_random_phrase


def test_generate_random_phrase_with_seeded_rng():
    """Test that the phrase generator produces deterministic output with a seeded RNG"""
    # Create two separate RNG instances with the same seed
    rng1 = random.Random(42)
    rng2 = random.Random(42)

    # Generate multiple phrases with each RNG
    phrases1: List[str] = [generate_random_phrase(rng1) for _ in range(10)]
    phrases2: List[str] = [generate_random_phrase(rng2) for _ in range(10)]

    # Verify phrases match between the two RNGs
    assert phrases1 == phrases2

    # Verify structure of generated phrases
    for phrase in phrases1:
        # Check phrase format (adjective-noun)
        assert len(phrase.split("-")) == 2
        adjective, noun = phrase.split("-")
        assert adjective in ADJECTIVES
        assert noun in NOUNS


def test_generate_random_phrase_without_rng():
    """Test that the phrase generator works with the default random module"""
    # Generate a phrase without providing an RNG
    phrase = generate_random_phrase()

    nouns = set()
    adjectives = set()
    phrases = set()
    for _ in range(100):
        phrase = generate_random_phrase()
        phrases.add(phrase)
        assert len(phrase.split("-")) == 2
        adjective, noun = phrase.split("-")
        nouns.add(noun)
        adjectives.add(adjective)

    assert len(nouns) > 1
    assert len(adjectives) > 1
    assert len(phrases) >= len(nouns) and len(phrases) >= len(adjectives)
