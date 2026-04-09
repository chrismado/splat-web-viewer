# Security Policy

## Reporting a Vulnerability

Please report vulnerabilities through a private GitHub security advisory or by contacting the maintainer directly before any public disclosure. Include reproduction steps, browser/runtime details, and any sample SPZ assets that trigger the issue.

## Audit Summary

- No hardcoded secrets or tokens were found during the April 2026 audit.
- No unsafe `eval()` usage, shell execution with user-controlled input, or server-side credential handling paths were identified.
- The Python WebRTC helper and TypeScript decoder paths should continue to be checked with `npm audit` and a source review for untrusted binary parsing changes.
