#!/bin/bash
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
cd /Users/dushyantkumar/Documents/BTP_ILAC_WAN
exec /opt/homebrew/bin/python3 -S -m http.server 5050 --directory web
