#!/bin/bash

# Script to create experience tags from config.yaml
# This script generates resim experience-tags create commands for all tags

echo "Creating experience tags..."

# Road type tags
./resim experience-tags create --name "road_type_single" --description "Tag for road type single"
./resim experience-tags create --name "road_type_interstate" --description "Tag for road type interstate"
./resim experience-tags create --name "road_type_rural" --description "Tag for road type rural"

# Weather tags
./resim experience-tags create --name "weather_rain" --description "Tag for weather rain"
./resim experience-tags create --name "weather_dry" --description "Tag for weather dry"
./resim experience-tags create --name "weather_fog" --description "Tag for weather fog"

# Lighting tags
./resim experience-tags create --name "lighting_morning" --description "Tag for lighting morning"
./resim experience-tags create --name "lighting_dusk" --description "Tag for lighting dusk"
./resim experience-tags create --name "lighting_day" --description "Tag for lighting day"
./resim experience-tags create --name "lighting_night" --description "Tag for lighting night"

echo "All experience tags created successfully!"
