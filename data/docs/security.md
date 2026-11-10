# Aurora Cloud Security Guide

## Authentication

All console users must enable multi-factor authentication within 7 days of account creation. Supported second factors are TOTP authenticator apps and FIDO2 hardware keys. SMS codes are not supported. Single sign-on through SAML 2.0 is available on the Business and Enterprise plans; SCIM user provisioning is Enterprise only.

## API keys

API keys are scoped to a single project and carry one of three roles: read-only, deployer, or admin. API keys expire after 90 days by default, and the maximum configurable lifetime is 365 days. A project can have at most 20 active keys. To rotate a key, create a replacement, deploy it, and then revoke the old key; a revoked key stops working within 60 seconds. Leaked keys detected by the secret scanner are revoked automatically and the project owners are emailed.

## Encryption

Data at rest is encrypted with AES-256. Data in transit requires TLS 1.2 or higher; TLS 1.0 and 1.1 are rejected. Customer-managed encryption keys are available on the Enterprise plan through the Key Vault service, and keys in Key Vault are rotated automatically every 180 days.

## Audit logs

Audit logs record every console and API action. Audit logs are retained for 90 days on the Business plan and 400 days on the Enterprise plan. The Starter plan does not include audit logs. Logs can be exported to a customer-owned storage bucket in JSON Lines format.

## Compliance

Aurora Cloud is certified for SOC 2 Type II and ISO 27001. A HIPAA business associate agreement is available to Enterprise customers only.
