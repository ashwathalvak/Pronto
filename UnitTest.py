import unittest
from Featurization import cleantext
from Featurization import removeStopwordswithStemming

class testTextCleaningFunctions(unittest.TestCase):
    def testCleaningFunction(self):
        self.assertEqual(cleantext("remove http:///www.google.com"),"remove |url|")
        self.assertEqual(cleantext("wayyyy tooooo tough"),"wayy too tough")

    def testStopwordRemovalFunction(self):
        self.assertEqual(removeStopwordswithStemming("generously"),"generous")
        self.assertEqual(removeStopwordswithStemming("having"),"")
        self.assertEqual(removeStopwordswithStemming("running"),"run")


if __name__ == '__main__':
    unittest.main()


def main():
    print("\n\n ------------------------------------------------------------------- ")
    print("\n \t\t Executing the Unit Test \n")
    print("------------------------------------------------------------------- \n")
    unittest.main()
    print("\n\n ------------------------------------------------------------------- ")
