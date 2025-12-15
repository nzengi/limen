//! Type definitions for the reputation system

use ark_ff::PrimeField;
use std::marker::PhantomData;

/// Fixed-point scale for representing fractional decay rate
pub const FIXED_POINT_SCALE: u64 = 1_000_000;

/// Default decay rate λ = 0.001 per epoch (slow decay)
/// Represented as fixed-point: 0.001 * 1,000,000 = 1,000
pub const DEFAULT_LAMBDA_FIXED: u64 = 1_000;

/// Maximum time delta supported in circuits (for efficiency)
pub const MAX_DELTA_T: usize = 365; // ~1 year in days

/// Event types that can affect exposure
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum EventType {
    /// No event, just time decay
    TimeDecay = 0,
    /// Negative event that increases exposure
    Penalty = 1,
    /// Positive action that decreases exposure
    Recovery = 2,
}

impl EventType {
    pub fn to_field<F: PrimeField>(self) -> F {
        F::from(self as u64)
    }

    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::TimeDecay),
            1 => Some(Self::Penalty),
            2 => Some(Self::Recovery),
            _ => None,
        }
    }
}

/// Complete reputation state (private)
/// This is the prover's internal state, never revealed publicly
#[derive(Clone, Debug)]
pub struct ExposureState<F: PrimeField> {
    /// Current exposure value (PRIVATE)
    /// Higher = worse reputation
    /// Must be non-negative
    pub exposure: F,

    /// Current timestamp/epoch number
    pub timestamp: F,

    /// Cryptographic commitment to entire history
    /// Binds all previous events together
    pub history_hash: F,

    /// Decay rate parameter λ ∈ (0,1)
    /// Stored as fixed-point value
    pub lambda: F,

    _phantom: PhantomData<F>,
}

impl<F: PrimeField> ExposureState<F> {
    /// Create genesis state (initial clean reputation)
    pub fn genesis(lambda: Option<F>) -> Self {
        let lambda =
            lambda.unwrap_or_else(|| F::from(DEFAULT_LAMBDA_FIXED) / F::from(FIXED_POINT_SCALE));

        Self {
            exposure: F::zero(),
            timestamp: F::zero(),
            history_hash: F::zero(), // Or hash of genesis parameters
            lambda,
            _phantom: PhantomData,
        }
    }

    /// Create state with specific values
    pub fn new(exposure: F, timestamp: F, history_hash: F, lambda: F) -> Self {
        Self {
            exposure,
            timestamp,
            history_hash,
            lambda,
            _phantom: PhantomData,
        }
    }

    /// Check if exposure is below threshold
    pub fn is_below_threshold(&self, threshold: F) -> bool {
        // Note: Field element comparison
        // This is a simplification; actual implementation needs careful handling
        self.exposure <= threshold
    }
}

/// Public commitment to state
/// This is what gets published/verified
#[derive(Clone, Debug, Default)]
pub struct StateCommitment<F: PrimeField> {
    /// Hash commitment to history (PUBLIC)
    pub history_hash: F,

    /// Current timestamp (PUBLIC)
    pub timestamp: F,

    /// Commitment to exposure value (PUBLIC)
    /// NOT the exposure itself, but a binding commitment
    /// Could be: Com(exposure, randomness)
    pub exposure_commitment: F,

    #[allow(dead_code)]
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> StateCommitment<F> {
    pub fn new(history_hash: F, timestamp: F, exposure_commitment: F) -> Self {
        Self {
            history_hash,
            timestamp,
            exposure_commitment,
            _phantom: PhantomData,
        }
    }

    /// Create commitment from state (includes commitment randomness)
    pub fn from_state(state: &ExposureState<F>, randomness: F) -> Self {
        // Pedersen commitment: Com(E, r) = E·G + r·H
        // Simplified here as just a hash for pseudocode
        let exposure_commitment = commit_value(state.exposure, randomness);

        Self::new(state.history_hash, state.timestamp, exposure_commitment)
    }
}

/// Input for a state transition
#[derive(Clone, Debug)]
pub struct TransitionInput<F: PrimeField> {
    /// Type of event
    pub event_type: EventType,

    /// Penalty amount (if Penalty event)
    pub penalty: F,

    /// Recovery amount (if Recovery event)
    pub recovery: F,

    /// Time elapsed since last update
    pub delta_t: u64,

    _phantom: PhantomData<F>,
}

impl<F: PrimeField> TransitionInput<F> {
    pub fn new(event_type: EventType, penalty: F, recovery: F, delta_t: u64) -> Self {
        Self {
            event_type,
            penalty,
            recovery,
            delta_t,
            _phantom: PhantomData,
        }
    }

    /// Create a time-decay-only input
    pub fn time_decay(delta_t: u64) -> Self {
        Self::new(EventType::TimeDecay, F::zero(), F::zero(), delta_t)
    }

    /// Create a penalty input
    pub fn penalty(amount: F, delta_t: u64) -> Self {
        Self::new(EventType::Penalty, amount, F::zero(), delta_t)
    }

    /// Create a recovery input
    pub fn recovery(amount: F, delta_t: u64) -> Self {
        Self::new(EventType::Recovery, F::zero(), amount, delta_t)
    }

    /// Convert to field elements for circuit
    pub fn to_fields(&self) -> Vec<F> {
        vec![
            self.event_type.to_field(),
            self.penalty,
            self.recovery,
            F::from(self.delta_t),
        ]
    }
}

/// Proof that exposure is below threshold
#[derive(Clone, Debug)]
pub struct ThresholdProof<F: PrimeField> {
    /// Compressed SNARK of the folded instance
    pub compressed_proof: Vec<u8>, // Placeholder for actual proof

    /// Final state commitment
    pub final_state: StateCommitment<F>,

    /// Public threshold value
    pub threshold: F,

    /// Public claim: is exposure < threshold?
    pub is_below_threshold: bool,

    #[allow(dead_code)]
    _phantom: PhantomData<F>,
}

impl<F: PrimeField> ThresholdProof<F> {
    pub fn new(
        compressed_proof: Vec<u8>,
        final_state: StateCommitment<F>,
        threshold: F,
        is_below_threshold: bool,
    ) -> Self {
        Self {
            compressed_proof,
            final_state,
            threshold,
            is_below_threshold,
            _phantom: PhantomData,
        }
    }
}

/// Helper function to commit to a value
/// In production: use Pedersen commitment
pub fn commit_value<F: PrimeField>(value: F, randomness: F) -> F {
    // Simplified: In real implementation, this would be:
    // value * G + randomness * H (elliptic curve points)
    // Here we just use a hash for pseudocode purposes
    hash_to_field(&[value, randomness])
}

/// Hash multiple field elements to a single field element
/// In production: use Poseidon hash
pub fn hash_to_field<F: PrimeField>(elements: &[F]) -> F {
    // Placeholder: actual implementation would use Poseidon
    // For now, simulate with a simple accumulation
    let mut result = F::zero();
    for (i, elem) in elements.iter().enumerate() {
        result += *elem * F::from((i + 1) as u64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;

    #[test]
    fn test_event_type_conversion() {
        assert_eq!(EventType::from_u64(1), Some(EventType::Penalty));
        assert_eq!(EventType::from_u64(99), None);
    }

    #[test]
    fn test_genesis_state() {
        let state: ExposureState<Fr> = ExposureState::genesis(None);
        assert_eq!(state.exposure, Fr::zero());
        assert_eq!(state.timestamp, Fr::zero());
    }

    #[test]
    fn test_transition_input_creation() {
        let input: TransitionInput<Fr> = TransitionInput::penalty(Fr::from(100u64), 30);
        assert_eq!(input.event_type, EventType::Penalty);
        assert_eq!(input.delta_t, 30);
    }
}
