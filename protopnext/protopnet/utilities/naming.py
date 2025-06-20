import random
import string

ADJECTIVES = [
    "active",
    "bouncy",
    "brave",
    "breezy",
    "bright",
    "brisk",
    "bubbly",
    "calm",
    "cheeky",
    "cheer",
    "chilly",
    "clever",
    "cozy",
    "crispy",
    "curvy",
    "dainty",
    "dandy",
    "dapper",
    "dreamy",
    "eager",
    "fair",
    "fancy",
    "feisty",
    "fiery",
    "fluffy",
    "gentle",
    "glad",
    "glossy",
    "graceful",
    "happy",
    "honest",
    "jazzy",
    "jolly",
    "jovial",
    "juicy",
    "jumpy",
    "kind",
    "lively",
    "lovely",
    "loyal",
    "mellow",
    "mighty",
    "neat",
    "nifty",
    "nimble",
    "peachy",
    "peppy",
    "perky",
    "plucky",
    "plush",
    "polite",
    "primal",
    "proud",
    "quick",
    "quiet",
    "quirky",
    "rapid",
    "rustic",
    "shiny",
    "shy",
    "silent",
    "simple",
    "sleek",
    "smooth",
    "snappy",
    "snug",
    "sober",
    "soft",
    "spicy",
    "spry",
    "spunky",
    "steady",
    "strong",
    "sturdy",
    "sunny",
    "swift",
    "tender",
    "thrifty",
    "tidy",
    "tiny",
    "tough",
    "true",
    "vivid",
    "warm",
    "wild",
    "witty",
    "zany",
    "zany",
    "zappy",
    "zesty",
    "zippy",
]
NOUNS = [
    "apple",
    "balloon",
    "banjo",
    "beach",
    "bridge",
    "castle",
    "chair",
    "cherry",
    "circle",
    "cloud",
    "cradle",
    "dream",
    "eagle",
    "field",
    "flame",
    "flower",
    "forest",
    "garden",
    "glove",
    "heart",
    "honey",
    "house",
    "island",
    "jewel",
    "kitten",
    "lemon",
    "meadow",
    "melon",
    "mirror",
    "morning",
    "mount",
    "ocean",
    "parrot",
    "pearl",
    "pebble",
    "phoenix",
    "planet",
    "plant",
    "plume",
    "puppy",
    "ribbon",
    "river",
    "rocket",
    "sapphire",
    "school",
    "sculpture",
    "season",
    "shadow",
    "shelter",
    "shore",
    "silver",
    "sister",
    "smile",
    "snow",
    "spark",
    "spirit",
    "stone",
    "stream",
    "sunset",
    "temple",
    "throne",
    "thunder",
    "train",
    "twilight",
    "unicorn",
    "valley",
    "vessel",
    "village",
    "vortex",
    "whale",
    "wildcat",
    "window",
    "zephyr",
]


def generate_random_phrase(rng: random.Random = random) -> str:
    """
    Generate a random adjective-noun phrase using a specified random number generator.

    Parameters:
        rng (random.Random, optional): A pseudorandom number generator instance.
        Defaults to `random` if not provided.

    Returns:
        str: A random phrase in the form of 'adjective-noun'.
    """
    adjective = rng.choice(ADJECTIVES)
    noun = rng.choice(NOUNS)
    return f"{adjective}-{noun}"


def k_letters(k: int, rng: random.Random = random) -> str:
    """
    Generate a random string of lowercase letters of length `k`.
    """
    return "".join(rng.choices(string.ascii_lowercase, k=k))
