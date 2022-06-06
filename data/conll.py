from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import tempfile
import subprocess
import operator
import collections

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))