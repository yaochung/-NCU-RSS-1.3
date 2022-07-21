import tensorflow as tf
import pycodestyle
import glob
import os


class TestCodeFormat(tf.test.TestCase):

    def test_conformance(self):
        """Test that we conform to PEP-8."""
        path = os.getcwd()
        parent_path = os.path.dirname(path)
        # cfg_path = os.path.join('src', 'test', 'setup.cfg')
        cfg_path = os.path.join('setup.cfg')
        style = pycodestyle.StyleGuide(config_file=cfg_path)
        # filenames = glob.glob(os.path.join(parent_path, 'src') + '\**\*.py', recursive=True)
        filenames = glob.glob(os.path.join(parent_path, 'src') + '/**/*.py', recursive=True)
        result = style.check_files(filenames)

        self.assertEqual(result.total_errors, 0, msg="Found code style errors (and warnings).")
