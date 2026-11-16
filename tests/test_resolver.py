import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from familiar.resolver import Resolver, extract_clues, find_addressed_name, find_self_name  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 3
        return self.t


def convo(*lines):
    r = Resolver(clock=Clock())
    for speaker, text in lines:
        r.feed(speaker, text)
    return r


def person(r, sid):
    return next(p for p in r.snapshot()["people"] if p["id"] == sid)


class SelfIntro(unittest.TestCase):
    def test_simple(self):
        for text in ["Hi, I'm Maya.", "Hi I’m Maya, nice to meet you", "My name is Maya", "Hey, this is Maya",
                     "Hello, it's Maya", "Call me Maya", "So I am Maya and I work here"]:
            self.assertEqual(find_self_name(text), "Maya", text)

    def test_two_part_name(self):
        self.assertEqual(find_self_name("My name is Maya Chen and I design hardware"), "Maya Chen")
        self.assertEqual(find_self_name("I'm Maya Chen."), "Maya Chen")

    def test_lowercase_after_strong_phrase(self):
        self.assertEqual(find_self_name("my name is priya"), "Priya")

    def test_not_names(self):
        for text in ["I'm good", "I'm fine, thanks", "I'm Tired today", "I'm here for the meetup", "I'm a designer",
                     "I'm Sorry about that", "It's Friday", "Hi, I'm in robotics", "This is a great event",
                     "Hi, welcome everyone", "I am Excited to be here"]:
            self.assertIsNone(find_self_name(text), text)

    def test_trailing_word_is_dropped(self):
        self.assertEqual(find_self_name("I'm Maya And I work here"), "Maya")


class Addressed(unittest.TestCase):
    def test_patterns(self):
        self.assertEqual(find_addressed_name("Nice to meet you, Dev."), "Dev")
        self.assertEqual(find_addressed_name("Great to meet you Dev"), "Dev")
        self.assertEqual(find_addressed_name("Thanks, Dev!"), "Dev")
        self.assertEqual(find_addressed_name("Hi Dev, over here"), "Dev")
        self.assertEqual(find_addressed_name("Over by the door. Hi Dev, good to see you again."), "Dev")
        self.assertIsNone(find_addressed_name("Hi, I'm Maya"))
        self.assertIsNone(find_addressed_name("Thanks for coming"))
        self.assertIsNone(find_addressed_name("Thanks Friday"))


class Conversation(unittest.TestCase):
    def test_self_intro_names_current_speaker(self):
        r = convo((0, "Hey, welcome to the meetup."), (1, "Thanks! I'm Maya, I work at Northline."))
        self.assertEqual(person(r, 1)["name"], "Maya")
        self.assertEqual(person(r, 1)["status"], "heard")
        self.assertIsNone(person(r, 0)["name"])
        self.assertIn("Works at Northline", person(r, 1)["clues"])

    def test_addressed_names_previous_speaker_as_possible(self):
        r = convo((1, "Do you know where the coffee is?"), (0, "Over there. Nice to meet you, Dev."))
        self.assertEqual(person(r, 1)["name"], "Dev")
        self.assertEqual(person(r, 1)["status"], "possible")
        self.assertIsNone(person(r, 0)["name"])

    def test_greeting_names_previous_speaker(self):
        r = convo((1, "Excuse me, is this seat taken?"), (0, "Hi Sam, no, go ahead."))
        self.assertEqual(person(r, 1)["name"], "Sam")

    def test_answer_to_name_question(self):
        r = convo((0, "I'm Alex. What's your name?"), (1, "Priya."))
        self.assertEqual(person(r, 1)["name"], "Priya")
        self.assertEqual(person(r, 1)["status"], "heard")
        self.assertEqual(person(r, 0)["name"], "Alex")

    def test_heard_beats_possible(self):
        r = convo((1, "Hello there."), (0, "Nice to meet you, Davin."), (1, "It's Dev, actually. My name is Dev."))
        self.assertEqual(person(r, 1)["name"], "Dev")
        self.assertEqual(person(r, 1)["status"], "heard")
        self.assertIn("Davin", person(r, 1)["alts"])

    def test_possible_does_not_replace_heard(self):
        r = convo((1, "Hi, I'm Maya."), (0, "Nice to meet you, Mia."))
        self.assertEqual(person(r, 1)["name"], "Maya")
        self.assertIn("Mia", person(r, 1)["alts"])

    def test_typed_name_wins(self):
        r = convo((1, "Hi, I'm Maya."))
        r.set_name(1, "maya chen")
        self.assertEqual(person(r, 1)["name"], "Maya Chen")
        self.assertEqual(person(r, 1)["status"], "confirmed")
        r.feed(1, "Actually my name is Mara.")
        self.assertEqual(person(r, 1)["name"], "Maya Chen")
        self.assertIn("Mara", person(r, 1)["alts"])

    def test_not_maya_clears_and_blocks_same_name(self):
        r = convo((1, "Hi, I'm Maya."))
        r.clear_name(1)
        self.assertIsNone(person(r, 1)["name"])
        self.assertEqual(person(r, 1)["status"], "unknown")
        r.feed(1, "Yes, Maya is my name, my name is Maya.")
        self.assertIsNone(person(r, 1)["name"])
        r.set_name(1, "Priya")
        self.assertEqual(person(r, 1)["name"], "Priya")

    def test_interim_never_names(self):
        r = Resolver(clock=Clock())
        r.feed(1, "hi I'm Maya", final=False)
        self.assertIsNone(r.snapshot()["people"][0]["name"])
        self.assertEqual(r.snapshot()["now"], 1)
        r.feed(1, "Hi, I'm Maya.", final=True)
        self.assertEqual(person(r, 1)["name"], "Maya")
        self.assertIsNone(r.snapshot()["interim"])

    def test_no_name_stays_unknown(self):
        r = convo((0, "How's the demo going?"), (1, "Pretty well, thanks."))
        self.assertEqual({p["status"] for p in r.snapshot()["people"]}, {"unknown"})

    def test_merge_split_voice(self):
        r = convo((1, "Hi, I'm Maya."), (2, "I work at Northline."), (0, "Great."))
        r.merge(2, 1)
        people = {p["id"]: p for p in r.snapshot()["people"]}
        self.assertNotIn(2, people)
        self.assertIn("Works at Northline", people[1]["clues"])
        r.feed(2, "I'm a product designer.")          # the old voice number still lands on the merged person
        self.assertIn("Product designer", person(r, 1)["clues"])

    def test_address_ignored_when_no_one_spoke_before(self):
        r = convo((0, "Hi Maya!"))
        self.assertIsNone(person(r, 0)["name"])

    def test_events_for_toast(self):
        r = convo((1, "Hi, I'm Maya."))
        events = r.snapshot()["events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["name"], "Maya")
        self.assertIn("I'm Maya", events[0]["why"])

    def test_me_flag(self):
        r = convo((0, "Hello everyone."), (1, "Hi."))
        r.set_me(0)
        self.assertEqual(person(r, 0)["display"], "You")
        self.assertFalse(person(r, 1)["is_me"])

    def test_missing_speaker_uses_last_voice(self):
        r = convo((1, "Hi, I'm Maya."))
        r.feed(None, "I work at Northline.")
        self.assertEqual(len(r.snapshot()["people"]), 1)


class Clues(unittest.TestCase):
    def test_clues(self):
        self.assertEqual(extract_clues("I work at Northline Labs."), ["Works at Northline Labs"])
        self.assertIn("Product designer", extract_clues("I'm a product designer at Northline"))
        self.assertIn("Works at Northline", extract_clues("I'm a product designer at Northline"))
        self.assertEqual(extract_clues("I'm from Oakland."), ["From Oakland"])
        self.assertIn("Studies robotics", extract_clues("I study robotics at Berkeley"))
        self.assertTrue(extract_clues("I'm working on a robot gripper and it's hard")[0].startswith("Building a robot gripper"))
        self.assertEqual(extract_clues("The weather is nice."), [])


if __name__ == "__main__":
    unittest.main()
