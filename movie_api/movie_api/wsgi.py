"""
WSGI config for movie_api project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os
import site
import sys
import posixpath

from django.core.wsgi import get_wsgi_application
import django.core.handlers.wsgi

site.addsitedir('~/.anaconda3/envs/REST/lib/python3.8/site-packages')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie_api.settings')

application = get_wsgi_application()
