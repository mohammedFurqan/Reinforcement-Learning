#!/usr/bin/env python
# -*- coding: utf-8 -*-


from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

