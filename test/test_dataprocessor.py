#coding=utf-8
import sys
sys.path.append("..")
import unittest

from data_processor.data_loader import DataProcessor

class TestProcessor(unittest.TestCase):
    def runTest(self):
        self.test_build_vocab("D:\\python script\\text_cnn\\data\\cnews.test.txt", "D:\\python script\\text_cnn\\data\\test_vocob.txt", 5000)
        return

    def test_build_vocab(self, train_path, vocob_path, vocob_size = 5000):
        data_processor = DataProcessor()
        data_processor.build_vocob(train_path, vocob_path, vocob_size)

if __name__ == "__main__":
    testcase = TestProcessor()
    testcase.runTest()