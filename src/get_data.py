import random


random.seed(95)  # ⚡


def adapt_or_die(word: str) -> str:
    # 0 -> change letter; 1 -> remove letter; 2 -> add letter
    action = random.randint(0, 2)
    res = ""

    for c in word:
        # this is a wild guess, but I am assuming that on average every 9th character is misspelled
        if random.randint(0, 8) == 1:
            if action == 0:
                res += chr(random.randint(4304, 4336))  # ord('ა') == 4304 and ord('ჰ') == 4336
            elif action == 1:
                continue
            else:
                res += c + chr(random.randint(4304, 4336))
        else:
            res += c

    return res


if __name__ == "__main__":
    for w in ["გამარჯობა", "ნიტა", "ლოყა", "პრინცესა", "სიხარული", "სიყვარული", "კეთილი", "ლამაზი", "საყვარელი", "ჭკვიანი", "ცხოვრება"]:
        print(adapt_or_die(w))