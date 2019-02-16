#!/bin/sh
python music_processor.py && python infer.py don && python infer.py ka && python synthesize.py 1
