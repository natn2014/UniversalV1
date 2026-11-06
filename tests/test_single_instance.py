import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.single_instance import acquire

HOLD = "import sys, time; sys.path.insert(0, '.'); from app.single_instance import acquire; " \
       "from pathlib import Path; h = acquire(Path(sys.argv[1])); print('held' if h else 'busy', flush=True); time.sleep(3)"


class SingleInstance(unittest.TestCase):
    def test_second_process_is_refused_until_first_exits(self):
        lock = Path(tempfile.mkdtemp()) / "test.lock"
        first = subprocess.Popen([sys.executable, "-c", HOLD, str(lock)], stdout=subprocess.PIPE, text=True)
        self.assertEqual(first.stdout.readline().strip(), "held")
        self.assertIsNone(acquire(lock))  # this process is "the second copy"
        first.kill()
        first.wait()
        handle = acquire(lock)  # released by the OS when the holder died
        self.assertIsNotNone(handle)
        handle.close()


if __name__ == "__main__":
    unittest.main()
