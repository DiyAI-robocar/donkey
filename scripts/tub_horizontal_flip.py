#!/usr/bin/python3

import sys
import glob
import json
from PIL import Image

if len(sys.argv) < 1:
  print("Usage: " + sys.argv[0] + " TUB_DIRECTORY")

def load_json(path):
  with open(path) as fd:
    return json.load(fd)

def save_json(path, obj):
  with open(path, 'w') as outfile:
    json.dump(obj, outfile)

def flip_angle(obj):
  obj["user/angle"] = -obj["user/angle"]
  return obj

def flip_image(jpg_name):
  Image.open(jpg_name).transpose(Image.FLIP_LEFT_RIGHT).save(jpg_name);

for filename in glob.iglob(sys.argv[1]+"/record_*.json"):
  print(filename)
  jpg_name = filename.replace('record_','').replace('.json', '_cam-image_array_.jpg')
  flip_image(jpg_name)
  save_json(filename, flip_angle(load_json(filename)))

