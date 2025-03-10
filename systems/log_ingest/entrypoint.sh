#!/bin/sh

echo "ReSim Log Ingestion Service"

source_dir="/tmp/resim/inputs"
dest_dir="/tmp/resim/outputs"

if [ -d "$source_dir" ]; then
    if [ "$(ls -A $source_dir)" ]; then
      if [ -d "$dest_dir" ]; then
          cp -r "$source_dir"/* "$dest_dir"
          echo "Files copied from $source_dir to $dest_dir"
      else
          echo "ReSim output directory $dest_dir does not exist. Have you mounted it?"
      fi
    else
      echo "ReSim input directory $source_dir is empty. Are you sure there is a log in the specified location?"
    fi
else
    echo "ReSim input directory $source_dir does not exist. have you mounted it?"
fi

echo "Finished Ingesting Logs"