import logging

# Attach a NullHandler so library users don’t get “No handlers could be found…”
# and so we don’t configure logging at import time.
logging.getLogger("solshade").addHandler(logging.NullHandler())
