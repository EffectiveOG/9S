#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create SSL directory if it doesn't exist
mkdir -p config/ssl

# Generate OpenSSL config file
cat > config/ssl/openssl.cnf << EOL
[ req ]
default_bits        = 4096
default_keyfile     = config/ssl/key.pem
distinguished_name  = req_distinguished_name
prompt             = no
encrypt_key        = no

[ req_distinguished_name ]
countryName            = "US"
stateOrProvinceName    = "California"
localityName           = "San Francisco"
organizationName       = "Jarvis Home Assistant"
organizationalUnitName = "Development"
commonName             = "localhost"
emailAddress          = "admin@localhost"
EOL

# Generate SSL certificate and key
openssl req \
    -x509 \
    -newkey rsa:4096 \
    -keyout config/ssl/key.pem \
    -out config/ssl/cert.pem \
    -days 365 \
    -nodes \
    -config config/ssl/openssl.cnf

# Check if certificate generation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}SSL certificates generated successfully!${NC}"
    echo "Certificate: config/ssl/cert.pem"
    echo "Private key: config/ssl/key.pem"
    
    # Set appropriate permissions
    chmod 600 config/ssl/key.pem
    chmod 644 config/ssl/cert.pem
    
    # Display certificate information
    echo -e "\nCertificate information:"
    openssl x509 -in config/ssl/cert.pem -text -noout | grep -E "Subject:|Issuer:|Not Before:|Not After :|Serial"
else
    echo -e "${RED}Failed to generate SSL certificates${NC}"
    exit 1
fi