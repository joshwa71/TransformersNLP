from transformers import pipeline, set_seed

import textwrap
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

lines = [line.rstrip() for line in open("./data/robert_frost.txt")]
lines = [line for line in lines if len(line) > 0]

gen = pipeline("text-generation")
set_seed(123)

def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

out = gen(lines[0], max_length=30)

prev =  out[0]['generated_text']

out = gen(prev + "\n" + lines[2], max_length=60)
print(wrap(out[0]['generated_text']))



