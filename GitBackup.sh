#!/bin/bash
# Created as part of the HPC maintenance, to avoid any dataloss whilst Git is not functional. 
# Define the destination directory
DESTINATION="/work3/s174159/git_backup_done"

# Make sure the destination directory exists
mkdir -p "$DESTINATION"

# Copy all tracked files to the destination
git ls-files | xargs -I {} cp --parents {} "$DESTINATION"

echo "All tracked files have been copied to $DESTINATION"
