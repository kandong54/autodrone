#!/bin/bash

# Based on https://github.com/improbable-eng/grpc-web/tree/master/misc

# Regenerate the self-signed certificate for local host. Recent versions of firefox and chrome(ium)
# require a certificate authority to be imported by the browser (localhostCA.pem) while
# the server uses a cert and key signed by that certificate authority.
# Based partly on https://stackoverflow.com/a/48791236
CA_PASSWORD=notsafe

# Generate the root certificate authority key with the set password
openssl genrsa -des3 -passout pass:$CA_PASSWORD -out cert/localhostCA.key 2048

# Generate a root-certificate based on the root-key for importing to browsers.
openssl req -x509 -new -nodes -key cert/localhostCA.key -passin pass:$CA_PASSWORD -config cert/localhostCA.conf -sha256 -days 90 -out cert/localhostCA.pem

# Generate a new private key
openssl genrsa -out cert/localhost.key 2048

# Generate a Certificate Signing Request (CSR) based on that private key (reusing the
# localhostCA.conf details)
openssl req -new -key cert/localhost.key -out cert/localhost.csr -config cert/localhostCA.conf

# Create the certificate for the webserver to serve using the localhost.conf config.
openssl x509 -req -in cert/localhost.csr -CA cert/localhostCA.pem -CAkey cert/localhostCA.key -CAcreateserial \
-out cert/localhost.crt -days 90 -sha256 -extfile cert/localhost.conf -passin pass:$CA_PASSWORD