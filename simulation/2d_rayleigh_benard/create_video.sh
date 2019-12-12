#!/bin/bash

ffmpeg -framerate 10 -pattern_type glob -i "frames/*.png" -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
