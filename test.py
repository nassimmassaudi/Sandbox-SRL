from hydra._internal.utils import _locate

import sys
print("PYTHONPATH:", sys.path)  # Output current Python path

# Try locating the class to see if it raises an error
try:
  cls = _locate("agents.drq_agent.DRQAgent")
  print("Class found:", cls)
except Exception as e:
  print("Error locating class:", e)