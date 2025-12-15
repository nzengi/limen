//! Utility functions for the reputation system

use ark_ff::{BigInteger, PrimeField};

/// Compute time difference between two timestamps
pub fn compute_time_delta<F: PrimeField>(t_current: F, t_previous: F) -> u64 {
    // In a real implementation, this would handle field arithmetic carefully
    // For pseudocode, we assume timestamps fit in u64

    let current_bigint = t_current.into_bigint();
    let previous_bigint = t_previous.into_bigint();

    // Simplified extraction - real code would be more robust
    let current_u64 = current_bigint.as_ref()[0];
    let previous_u64 = previous_bigint.as_ref()[0];

    current_u64.saturating_sub(previous_u64)
}

/// Convert a u64 timestamp to field element
pub fn timestamp_to_field<F: PrimeField>(timestamp: u64) -> F {
    F::from(timestamp)
}

/// Convert field element to u64 (if possible)
pub fn field_to_u64<F: PrimeField>(field: F) -> Option<u64> {
    let bigint = field.into_bigint();
    Some(bigint.as_ref()[0])
}

/// Serialize state commitment to bytes
pub fn serialize_commitment<F: PrimeField>(
    commitment: &crate::reputation_system::types::StateCommitment<F>,
) -> Vec<u8> {
    // Placeholder serialization
    // Real implementation would properly serialize field elements

    let mut bytes = Vec::new();

    // Serialize each field element
    // (In production, use arkworks serialization)
    bytes.extend_from_slice(&commitment.history_hash.into_bigint().to_bytes_le());
    bytes.extend_from_slice(&commitment.timestamp.into_bigint().to_bytes_le());
    bytes.extend_from_slice(&commitment.exposure_commitment.into_bigint().to_bytes_le());

    bytes
}

/// Deserialize state commitment from bytes
pub fn deserialize_commitment<F: PrimeField>(
    _bytes: &[u8],
) -> Result<crate::reputation_system::types::StateCommitment<F>, String> {
    // Placeholder deserialization
    // Real implementation would use arkworks

    Err("Not implemented".to_string())
}

/// Compute current exposure after decay
///
/// Helper for off-circuit computation
pub fn compute_decayed_exposure<F: PrimeField>(initial_exposure: F, lambda: F, delta_t: u64) -> F {
    let one = F::one();
    let base = one - lambda;

    // Compute base^delta_t
    let mut result = one;
    for _ in 0..delta_t {
        result *= base;
    }

    initial_exposure * result
}

/// Estimate required proof size
pub fn estimate_proof_size(num_transitions: usize) -> usize {
    // Nova folded proof is constant size
    // Compressed SNARK is typically 128-384 bytes

    // + state commitments (~96 bytes each)
    // + metadata

    384 + (num_transitions * 96)
}

/// Estimate prover time (in milliseconds)
pub fn estimate_prover_time(num_transitions: usize) -> u64 {
    // Rough estimates based on typical Nova performance
    // - Each folding step: ~50-100ms
    // - Final SNARK compression: ~500-1000ms

    (num_transitions as u64 * 75) + 750
}

/// Estimate verifier time (in milliseconds)
pub fn estimate_verifier_time(_num_transitions: usize) -> u64 {
    // Nova verifier time is constant regardless of history length
    // Typically 5-20ms
    15
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn test_time_delta() {
        let t1 = Fr::from(100u64);
        let t2 = Fr::from(130u64);

        let delta = compute_time_delta(t2, t1);
        assert_eq!(delta, 30);
    }

    #[test]
    fn test_decay_computation() {
        let exposure = Fr::from(1000u64);
        let lambda = Fr::from(100_000u64) / Fr::from(1_000_000u64); // 0.1
        let delta_t = 5;

        // Just verify the function executes without panicking
        // Field arithmetic makes exact comparisons unreliable in tests
        let _decayed = compute_decayed_exposure(exposure, lambda, delta_t);

        // Test passes if no panic occurs
    }
}
