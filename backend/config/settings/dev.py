# ruff: noqa: F403, F405
from .base import *  # noqa: F403

DEBUG = True

ALLOWED_HOSTS = ["*"]

SECRET_KEY = "dev-insecure-secret-key"

CORS_ALLOW_ALL_ORIGINS = True

SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

# Release each connection as soon as its work is done instead of keeping it warm
# for 60s. Production's keep-alive (see `base.py`) is a deliberate trade: holding
# connections open avoids re-authenticating on every request. It is the wrong
# trade for a dev or test process, where a connection opened by one worker thread
# would outlive the run and leave a live session on the test database — which then
# blocks Django from dropping that database at the end of the session.
DATABASES["default"]["CONN_MAX_AGE"] = 0
DATABASES["default"]["CONN_HEALTH_CHECKS"] = False
