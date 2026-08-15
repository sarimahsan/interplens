# Security Policy

## Supported Versions

InterpLens prioritizes security and stability. Critical security vulnerabilities will be addressed promptly for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Reporting a Vulnerability

We take the security of **InterpLens** seriously. If you discover a security issue or vulnerability (e.g. arbitrary code execution via untrusted model weights, unsafe deserialization, or network exposure risks), please report it responsibly:

1. **Do not create a public issue** on GitHub for sensitive security vulnerabilities.
2. **Email the maintainers directly** or use GitHub's private vulnerability reporting feature under the repository's **Security** tab.
3. Include details of the vulnerability:
   - Affected InterpLens version(s)
   - Python and PyTorch environments
   - Step-by-step reproduction instructions or a minimal Proof of Concept (PoC) script
   - Potential impact of the issue

---

## Security Best Practices for Users

- **Untrusted Models:** Only load PyTorch checkpoints and HuggingFace weights from trusted repositories and verified authors.
- **Remote Host Binding:** When binding InterpLens to `0.0.0.0` on remote cloud VMs, ensure traffic is routed through a secure reverse proxy (e.g. Nginx, Caddy) or protected behind a VPN/firewall.
- **API Token Security:** Never commit HuggingFace user tokens (`hf_...`) to public repositories or shared notebook outputs. Use the `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN` environment variables.

---

Thank you for helping keep the InterpLens open-source ecosystem safe!
