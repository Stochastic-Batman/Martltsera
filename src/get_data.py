import json
import os
import random


random.seed(95)  # ⚡

ALL_GEORGIAN_CHARS = [chr(i) for i in range(4304, 4337)]

# standard Georgian QWERTY keyboard neighbors to simulate realistic finger slips
KEYBOARD_NEIGHBORS: dict[str, list[str]] = {
    'ა': ['ს', 'ქ', 'ზ'],
    'ბ': ['ვ', 'გ', 'ჰ', 'ნ'],
    'გ': ['ფ', 'ტ', 'ყ', 'ჰ', 'ბ', 'ვ'],
    'დ': ['ს', 'ე', 'რ', 'ფ', 'ც', 'ხ'],
    'ე': ['წ', 'რ', 'დ', 'ს'],
    'ვ': ['ც', 'ფ', 'გ', 'ბ'],
    'ზ': ['ა', 'ს', 'ხ'],
    'თ': ['ღ', 'ყ', 'გ', 'ფ'],  # whenever we use "Shift +", all the neighbors are also with "Shift +" (if there exists one)
    'ი': ['უ', 'ო', 'ჯ', 'ჰ'],
    'კ': ['ჯ', 'ი', 'ო', 'ლ', 'მ'],
    'ლ': ['კ', 'მ', 'ო', 'პ'],
    'მ': ['ნ', 'ჯ', 'კ', 'ლ'],
    'ნ': ['ბ', 'ჰ', 'ჯ', 'მ'],
    'ო': ['ი', 'პ', 'ლ', 'კ', 'ჯ'],
    'პ': ['ო', 'ლ'],
    'ჟ': ['ჰ', 'უ', 'ი', 'კ', 'მ', 'ნ'],
    'რ': ['ე', 'ტ', 'ფ', 'დ'],
    'ს': ['ა', 'წ', 'ე', 'დ', 'ხ', 'ზ'],
    'ტ': ['რ', 'ყ', 'გ', 'ფ'],
    'უ': ['ყ', 'ი', 'ჯ', 'ჰ'],
    'ფ': ['დ', 'რ', 'ტ', 'გ', 'ვ', 'ც'],
    'ქ': ['წ', 'ა'],
    'ღ': ['ე', 'თ', 'ფ', 'დ'],
    'ყ': ['თ', 'უ', 'ჰ', 'გ'],
    'შ': ['ა', 'წ', 'ე', 'დ', 'ხ', 'ზ'],
    'ჩ': ['ხ', 'დ', 'ფ', 'ვ'],
    'ც': ['ხ', 'დ', 'ფ', 'ვ'],
    'ძ': ['ა', 'ს', 'ხ'],
    'წ': ['ქ', 'ე', 'ს', 'ა'],
    'ჭ': ['ქ', 'ე', 'ს', 'ა'],
    'ხ': ['ა', 'ს', 'დ', 'ც'],
    'ჯ': ['ჰ', 'უ', 'ი', 'კ', 'მ', 'ნ'],
    'ჰ': ['გ', 'ყ', 'უ', 'ჯ', 'ნ', 'ბ']
}


def read_sources(path: str) -> list[str]:
    data_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    res: list[str] = []

    for i in range(3):
        file_path = os.path.join(data_src_path, f"wordsChunk_{i}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            res.extend(content)

    return res


def get_typo_char(c: str) -> str:
    # 95% chance to pick a neighbor, 5% random char
    if c in KEYBOARD_NEIGHBORS and random.random() > 0.05:
        return random.choice(KEYBOARD_NEIGHBORS[c])
    return random.choice(ALL_GEORGIAN_CHARS)


def adapt_or_die(word: str) -> str:
    res = ""
    error_made = False

    # higher chance of error per char for shorter words otherwise short words often remain untouched
    prob = 0.15 if len(word) < 7 else 0.10

    for c in word:
        # ord('ა') == 4304 and ord('ჰ') == 4336
        if not (4304 <= ord(c) <= 4336):
            res += c
            continue

        if random.random() < prob:
            # 0 -> substitution p = 0.6; 1 -> deletion p = 0.2; 2 -> insertion p = 0.2
            action_roll = random.random()
            error_made = True

            if action_roll < 0.6:
                res += get_typo_char(c)  # substitution jutsu!
            elif action_roll < 0.8:
                continue  # deletion
            else:
                res += c + get_typo_char(c)  # insertion
        else:
            res += c

    # ensure the word is actually corrupted for training purposes
    if not error_made and len(res) > 0:
        idx = random.randint(0, len(res) - 1)
        temp_list = list(res)
        if 4304 <= ord(temp_list[idx]) <= 4336:
            temp_list[idx] = get_typo_char(temp_list[idx])
        res = "".join(temp_list)

    return res


def get_dataset_pairs(path: str = "../data_src", dictionary_size: int = 50000) -> list[tuple[str, str]]:
    pure_data = read_sources(path)
    random.shuffle(pure_data)

    dataset = []
    count = 0

    for word in pure_data:
        if count >= dictionary_size:
            break

        if len(word) < 3:  # skip extremely short words
            continue

        dataset.append((adapt_or_die(word), word))
        count += 1

    return dataset


if __name__ == "__main__":
    print("<== Testing adapt_or_die ==>")
    test_words = ["გამარჯობა", "გაგიმარჯოს", "ლადო", "ბრიუს ვეინი", "კომპიუტერი", "სიყვარული", "მაყუთი", "წიგნიერება", "გოკუ", "ტელეფონი", "ბეტმენი"]
    for w in test_words:
        print(f"{w} -> {adapt_or_die(w)}")

    print("\n<== Testing dataset generation ==>")
    data = get_dataset_pairs(dictionary_size=15)
    for corrupted, original in data:
        print(f"Corrupted: {corrupted} | Original: {original}")