#!/bin/bash

protein_name="Retinoblastoma"

# Format the protein name for the URL
formatted_protein_name=$(echo "$protein_name" | sed 's/ /+/g')

# Make the API request to search for the protein name
response=$(curl -s "https://www.uniprot.org/uniprotkb/?query=$formatted_protein_name&sort=score")

# Extract the Uniprot ID from the response
uniprot_id=$(echo "$response" | grep -oP 'href="/uniprot/(\w+)"' | grep -oP '(\w+)"$' | head -1)

# Output the Uniprot ID
echo "Uniprot ID for $protein_name: $uniprot_id"