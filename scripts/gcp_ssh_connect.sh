#!/bin/bash

# Required parameters
GCP_PROJECT_ID="asims-project"
GCP_ZONE="us-central1-b"              # Change to your instance's zone
GCP_INSTANCE_NAME="xsir-watermarking"
SSH_USER="amohamed"              # Optional: often your local username or set by gcloud config

# Authenticate if not already
gcloud auth login

# Set the project and zone (optional but helpful)
gcloud config set project "$GCP_PROJECT_ID"
gcloud config set compute/zone "$GCP_ZONE"

# SSH into the instance
gcloud compute ssh "$SSH_USER@$GCP_INSTANCE_NAME" \
    --project="$GCP_PROJECT_ID" \
    --zone="$GCP_ZONE"
