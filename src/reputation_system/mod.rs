//! Zero-Knowledge Time-Decaying Exposure Reputation System
//! 
//! This module provides a privacy-preserving reputation system where:
//! - Exposure increases on negative events (penalties)
//! - Exposure decays exponentially over time
//! - Users can prove exposure < threshold without revealing exact value
//! - Uses folding-based recursive ZK (Nova) with transparent setup

pub mod types;
pub mod circuits;
pub mod prover;
pub mod verifier;
pub mod utils;

// Re-export main types
pub use types::{ExposureState, StateCommitment, TransitionInput, EventType};
pub use prover::ReputationProver;
pub use verifier::ReputationVerifier;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_flow() {
        // This would test a basic reputation flow
        // penalty → decay → threshold check
    }
}
