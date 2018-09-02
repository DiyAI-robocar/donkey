#!/usr/bin/python3

# -*- coding: utf-8 -*-
import re
import sys

from donkeycar.management.base import execute_from_command_line

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    execute_from_command_line()

