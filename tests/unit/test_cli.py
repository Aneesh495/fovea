"""
Unit tests for the Fovea command-line interface.
"""
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from fovea.cli.main import main

class TestCLI(unittest.TestCase):
    def test_cli_help(self):
        with self.assertRaises(SystemExit) as cm:
            with patch("sys.stdout", new=io.StringIO()):
                main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_cli_inspect_stage(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            main(["inspect", "dialect"])
            output = fake_out.getvalue()
            self.assertIn("irdl.dialect @fovea", output)

if __name__ == "__main__":
    unittest.main()
