#!/bin/bash

# Script to download SARAH3 data
#
# First signup and request data chunks from
# https://wui.cmsaf.eu/safira/action/viewDoiDetails?acronym=SARAH_V003
#
# To use the script:
#
# Make it executable: chmod +x wget_script1.sh
# Run it: ./wget_script1.sh
#
# The script will:
#
# Download tar files to OUTPUT_DIR (defined below)
# Download recursively while ignoring parent directories
# Skip index.html files
# Retry up to 3 times if downloads fail
# Log all activity to LOG_FILE (defined below)

# How to extract data from .tar file:
#
# tar -xf filename.tar
# tar -xf $file
# for num in {1..9}; do tar -xf ORD12345/ORD12345_0$num\.tar; done
# 
# CHECK Disk Space with:
# quotas

# Configuration for CMSAF server
# Using info from CMSAF email (response to request for data)
USERNAME="abcde"
PASSWORD="abcde"

# Use order number from CMSAF email (response to request for data)
SERVER="https://cmsaf.dwd.de/data/"
ORDERNUM="ORD12345/"

SERVER_URL=$SERVER$ORDERNUM

# ------------

OUTPUT_DIR="/"
LOG_FILE="cmsaf_download_log.txt"
RETRY_ATTEMPTS=3
TIMEOUT=60

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "CMSAF download started at $(date)" > "$LOG_FILE"

# Function to handle download with retry
download_with_retry() {
    local attempt=1
    
    while [ $attempt -le $RETRY_ATTEMPTS ]; do
        echo "Download attempt $attempt of $RETRY_ATTEMPTS"
        
        wget \
            -r \
            -np \
            -nH \
            --cut-dirs=1 \
            --reject="index.html" \
            --timeout="$TIMEOUT" \
            --tries=1 \
            --user="$USERNAME" \
            --password="$PASSWORD" \
            --directory-prefix="$OUTPUT_DIR" \
            --append-output="$LOG_FILE" \
            --show-progress \
            "$SERVER_URL"

        if [ $? -eq 0 ]; then
            echo "Download completed successfully" >> "$LOG_FILE"
            return 0
        else
            echo "Attempt $attempt failed" >> "$LOG_FILE"
            ((attempt++))
            sleep 10  # Wait 10 seconds before retry
        fi
    done

    echo "Failed to download after $RETRY_ATTEMPTS attempts" >> "$LOG_FILE"
    return 1
}

# Main execution
echo "Starting CMSAF data download..."
echo "Data will be saved to: $OUTPUT_DIR"
echo "Check $LOG_FILE for progress details"

download_with_retry

# Final status
echo "Download process completed at $(date)" >> "$LOG_FILE"
echo "See $LOG_FILE for complete details"
