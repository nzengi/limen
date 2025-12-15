# Contributing to ZK Time-Decaying Exposure Reputation System

Thank you for your interest in contributing! This project is a research prototype exploring novel zero-knowledge proof systems for privacy-preserving reputation verification.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Research Contributions](#research-contributions)

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- ✅ Be respectful and constructive in discussions
- ✅ Welcome newcomers and help them get started
- ✅ Focus on what is best for the community and research
- ❌ Avoid personal attacks or unconstructive criticism
- ❌ Do not harass or discriminate against others

## 🚀 Getting Started

### Prerequisites

- **Rust**: 1.75.0 or later ([install](https://rustup.rs/))
- **Cargo**: Comes with Rust
- **Git**: For version control
- **Basic knowledge**: Zero-knowledge proofs, R1CS, cryptography (helpful but not required)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/nzengi/limen.git
cd limen

# Build the project
cargo build

# Run tests
cargo test

# Run the example
cargo run --example basic_reputation_flow
```

## 🔧 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/limen.git
cd limen

# Add upstream remote
git remote add upstream https://github.com/nzengi/limen.git
```

### 2. Create a Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
```

### 3. Install Development Tools

```bash
# Install rustfmt for code formatting
rustup component add rustfmt

# Install clippy for linting
rustup component add clippy

# Optional: Install cargo-watch for auto-rebuilding
cargo install cargo-watch
```

## 💡 How to Contribute

### Types of Contributions

1. **🐛 Bug Fixes**: Fix issues in existing code
2. **✨ New Features**: Add new functionality or improvements
3. **📝 Documentation**: Improve README, code comments, or technical docs
4. **🔬 Research**: Contribute cryptographic analysis or optimizations
5. **🧪 Tests**: Add or improve test coverage
6. **⚡ Performance**: Optimize circuit constraints or prover efficiency

### Finding Issues to Work On

- Check [open issues](https://github.com/nzengi/limen/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Ask in discussions if you're unsure where to start

### Before You Start

1. **Check existing issues/PRs**: Avoid duplicate work
2. **Discuss major changes**: Open an issue first for significant changes
3. **Read the design doc**: Review `time_decay_reputation_design.md`

## 📐 Code Style Guidelines

### Rust Style

Follow the official [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/):

```bash
# Format your code before committing
cargo fmt

# Check for common mistakes
cargo clippy -- -D warnings
```

### Naming Conventions

- **Types/Structs**: `PascalCase` (e.g., `ExposureState`, `ReputationProver`)
- **Functions/Methods**: `snake_case` (e.g., `apply_transition`, `prove_threshold`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_DECAY_RATE`)
- **Modules**: `snake_case` (e.g., `reputation_system`)

### Documentation

- **Public APIs**: Must have doc comments (`///`)
- **Complex logic**: Add inline comments explaining cryptographic reasoning
- **Examples**: Include usage examples in doc comments

```rust
/// Applies a state transition to the exposure value.
///
/// # Arguments
/// * `transition` - The transition input (penalty, recovery, or decay)
///
/// # Returns
/// * `Result<(), ReputationError>` - Success or error
///
/// # Example
/// ```
/// let penalty = TransitionInput::penalty(Fr::from(100u64), 0);
/// prover.apply_transition(penalty)?;
/// ```
pub fn apply_transition(&mut self, transition: TransitionInput<F>) -> Result<(), ReputationError> {
    // Implementation
}
```

### Code Organization

```
src/
├── lib.rs                      # Public API exports
└── reputation_system/
    ├── mod.rs                  # Module organization
    ├── types.rs                # Core types and structs
    ├── circuits.rs             # R1CS constraints
    ├── prover.rs               # Prover logic
    ├── verifier.rs             # Verifier logic
    └── utils.rs                # Helper functions
```

## 🧪 Testing Requirements

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run tests with coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html
```

### Test Coverage Requirements

- ✅ **Unit tests**: For all public functions
- ✅ **Integration tests**: For complete workflows
- ✅ **Edge cases**: Boundary conditions, overflows, invalid inputs
- ✅ **Cryptographic properties**: Soundness, completeness, zero-knowledge

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_penalty_increases_exposure() {
        let mut state = ExposureState::<Fr>::genesis(None);
        let penalty = TransitionInput::penalty(Fr::from(100u64), 0);
        
        // Apply transition
        state = state.apply_transition(&penalty).unwrap();
        
        // Verify exposure increased
        assert!(state.exposure > Fr::from(0u64));
    }

    #[test]
    #[should_panic(expected = "Invalid timestamp")]
    fn test_invalid_timestamp_panics() {
        let state = ExposureState::<Fr>::genesis(None);
        let invalid = TransitionInput::penalty(Fr::from(100u64), u64::MAX);
        state.apply_transition(&invalid).unwrap();
    }
}
```

## 🔄 Pull Request Process

### 1. Prepare Your Changes

```bash
# Ensure code is formatted
cargo fmt

# Check for warnings
cargo clippy -- -D warnings

# Run all tests
cargo test

# Update documentation if needed
cargo doc --no-deps --open
```

### 2. Commit Your Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add time decay constraint validation"
git commit -m "Fix overflow in exposure calculation"
git commit -m "Improve documentation for threshold proofs"

# Bad commit messages (avoid these)
git commit -m "fix bug"
git commit -m "update code"
git commit -m "changes"
```

### 3. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a pull request on GitHub
# Use the PR template and fill in all sections
```

### 4. PR Review Process

- **Automated checks**: CI will run tests and linting
- **Code review**: Maintainers will review your code
- **Feedback**: Address any requested changes
- **Approval**: Once approved, your PR will be merged

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines (`cargo fmt`, `cargo clippy`)
- [ ] All tests pass (`cargo test`)
- [ ] New tests added for new functionality
- [ ] Documentation updated (README, code comments)
- [ ] No new warnings introduced
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes

## 🔬 Research Contributions

### Cryptographic Analysis

If you're contributing cryptographic research:

1. **Formal analysis**: Provide mathematical proofs or security arguments
2. **References**: Cite relevant papers and prior work
3. **Assumptions**: Clearly state cryptographic assumptions
4. **Limitations**: Document any known limitations or attack vectors

### Optimization Ideas

For performance improvements:

1. **Benchmarks**: Provide before/after benchmarks
2. **Trade-offs**: Explain any security/performance trade-offs
3. **Constraint count**: Document impact on circuit size
4. **Prover time**: Measure impact on proof generation time

### Example Research Contribution

```markdown
## Optimized Decay Gadget

**Motivation**: Current exponential decay uses O(log Δt) constraints. 
This PR implements a lookup-table approach reducing to O(1) constraints.

**Security**: Maintains same security assumptions (discrete log hardness).
Lookup table is public and verifiable.

**Performance**: 
- Before: ~150 constraints per decay
- After: ~20 constraints per decay
- Prover time: 45ms → 12ms (3.75x faster)

**References**:
- [Lookup Arguments Paper](https://eprint.iacr.org/2020/315)
- [Plookup Implementation](https://github.com/example)
```

## 📚 Additional Resources

- **Technical Design**: [`time_decay_reputation_design.md`](time_decay_reputation_design.md)
- **Example Usage**: [`examples/basic_reputation_flow.rs`](examples/basic_reputation_flow.rs)
- **arkworks Docs**: [https://docs.rs/ark-r1cs-std/](https://docs.rs/ark-r1cs-std/)
- **Nova Paper**: [https://eprint.iacr.org/2021/370](https://eprint.iacr.org/2021/370)

## 💬 Getting Help

- **Questions**: Open a [discussion](https://github.com/nzengi/limen/discussions)
- **Bugs**: Create an [issue](https://github.com/nzengi/limen/issues)
- **Research**: Use the "Research Discussion" issue template

## 🙏 Recognition

Contributors will be:

- Listed in our CONTRIBUTORS.md file
- Credited in release notes for significant contributions
- Acknowledged in academic papers (if applicable)

---

Thank you for contributing to advancing privacy-preserving reputation systems! 🚀

**Last Updated**: 2025-12-15
