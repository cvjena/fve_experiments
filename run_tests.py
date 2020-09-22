#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import sys
import unittest

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tests import *

logging.disable(logging.CRITICAL)
unittest.main()
