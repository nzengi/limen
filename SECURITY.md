# Security Policy

## 🔐 Reporting a Vulnerability

We take the security of the ZK Time-Decaying Exposure Reputation System seriously. If you discover a security vulnerability, please follow these guidelines:

### ⚠️ **DO NOT** create a public GitHub issue for security vulnerabilities

Instead, please report security issues privately:

1. **Email**: Send details to [howyaniii@gmail.com] with subject line: `[SECURITY] limen Vulnerability`
2. **Expected Response Time**: We aim to acknowledge receipt within 48 hours
3. **Disclosure Timeline**: We will work with you to understand and address the issue within 90 days

### 📋 What to Include in Your Report

Please provide as much information as possible:

- **Type of vulnerability** (e.g., cryptographic flaw, implementation bug, side-channel attack)
- **Affected component** (e.g., circuit constraints, prover logic, verifier)
- **Steps to reproduce** the vulnerability
- **Potential impact** and severity assessment
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up questions

### 🎯 Scope

This security policy applies to:

- ✅ Cryptographic protocol design
- ✅ Circuit constraint implementations
- ✅ Prover and verifier logic
- ✅ Commitment schemes and hash functions
- ✅ Side-channel vulnerabilities
- ✅ Soundness and completeness properties

### 🏆 Recognition

We appreciate responsible disclosure and will:

- Acknowledge your contribution in our security advisories (with your permission)
- Credit you in our CHANGELOG for security fixes
- Work collaboratively to understand and resolve the issue

## 🛡️ Supported Versions

| Version | Status | Support |
|---------|--------|---------|
| main (latest) | 🔬 Research | Active development |
| v0.1.x | 🧪 Prototype | Security reports accepted |

> **⚠️ Important**: This is a **research prototype**. It is **NOT production-ready** and should not be used in production systems without:
> - Formal security audit by cryptography experts
> - Complete implementation of Nova folding schemes
> - Comprehensive testing and fuzzing
> - Side-channel attack mitigation

## 🔍 Known Limitations

We are aware of the following limitations:

1. **Research-Level Code**: This is a proof-of-concept implementation
2. **Incomplete Nova Integration**: Full folding scheme integration is planned
3. **No Formal Verification**: Constraints have not been formally verified
4. **Performance Optimizations Needed**: Not optimized for production use
5. **Side-Channel Attacks**: No protection against timing or power analysis attacks

## 🔐 Security Best Practices

If you're experimenting with this code:

### For Researchers
- ✅ Use this for academic research and experimentation
- ✅ Cite security assumptions clearly in your work
- ✅ Validate cryptographic properties independently
- ❌ Do not deploy in production without extensive auditing

### For Developers
- ✅ Review the constraint system carefully
- ✅ Test with malicious inputs and edge cases
- ✅ Use constant-time operations where applicable
- ✅ Validate all public inputs
- ❌ Do not trust unaudited cryptographic implementations

### For Auditors
- 🔍 Focus on constraint completeness and soundness
- 🔍 Check for arithmetic overflows and underflows
- 🔍 Verify range checks and boundary conditions
- 🔍 Analyze potential side-channel leakage
- 🔍 Review commitment scheme security

## 📚 Cryptographic Assumptions

This system relies on:

1. **Discrete Logarithm Problem**: Pedersen commitments security
2. **Collision Resistance**: Poseidon hash function
3. **R1CS Soundness**: Constraint system completeness
4. **Nova Security**: Folding scheme assumptions (when integrated)

## 🔗 Security Resources

- [Nova Paper](https://eprint.iacr.org/2021/370) - Folding scheme security analysis
- [Poseidon Paper](https://eprint.iacr.org/2019/458) - Hash function security
- [arkworks Security](https://github.com/arkworks-rs/algebra) - Underlying library security

## 📝 Security Changelog

### [Unreleased]
- Initial research implementation
- Basic constraint system
- Prototype prover/verifier logic

---

**Last Updated**: 2025-12-15

For non-security related issues, please use the [GitHub issue tracker](https://github.com/nzengi/limen/issues).
