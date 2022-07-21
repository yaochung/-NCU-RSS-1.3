import json
import os
import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        cfg_path = os.path.join('coverage.json')
        f = open(cfg_path)
        coverage = json.load(f)
        self.assertGreater(int(coverage['totals']['percent_covered_display']), 85,
                           "coverage less than expected")  # add assertion here
