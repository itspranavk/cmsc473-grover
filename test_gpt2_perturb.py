import unittest
from gpt2_sent import replaceQuotes, joinSentences, getGPT2
class TestGPT2Perturb(unittest.TestCase):

    def test_replaceQuotes(self):
        self.assertEqual(replaceQuotes("testString"), "testString")
        self.assertEqual(replaceQuotes(""), "")
        self.assertEqual(replaceQuotes("test'String"), "test\u2019String")
        self.assertEqual(replaceQuotes('test"String"'), "test\u201cString\u201d")
        self.assertEqual(replaceQuotes('test"String'), "test\u201cString")
        self.assertEqual(replaceQuotes('""""'), "\u201c\u201d\u201c\u201d")
        self.assertEqual(replaceQuotes('"""'), "\u201c\u201d\u201c")

    def test_joinSentences(self):
        test_sentences = ["Test1.", "Test2."]
        self.assertEqual(joinSentences([1,1], test_sentences), "Test1.\nTest2.\n")
        self.assertEqual(joinSentences([2], test_sentences), "Test1. Test2.\n")
        self.assertEqual(joinSentences([], []), "")

    def test_getGPT2(self):
        self.assertNotEqual(getGPT2("Test text."), None)
        self.assertNotEqual(len(getGPT2("Other text.")), 0)
        with self.assertRaises(SystemExit):
            getGPT2("")

if __name__ == "__main__":
    unittest.main()