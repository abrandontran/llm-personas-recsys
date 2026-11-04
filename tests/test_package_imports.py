import unittest

class TestPackageImports(unittest.TestCase):
    def test_package_structure(self):
        import src
        import src.data
        import src.envs
        import src.models
        import src.metrics
        self.assertTrue(hasattr(src, "__version__"))
        self.assertEqual(src.__version__, "0.1.0")

if __name__ == "__main__":
    unittest.main()
