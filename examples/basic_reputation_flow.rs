//! Example: Basic reputation flow
//!
//! Demonstrates:
//! 1. System initialization
//! 2. Applying penalties
//! 3. Time decay
//! 4. Recovery actions
//! 5. Threshold proof generation and verification

use ark_bn254::Fr;
use zk_algorithm::reputation_system::utils;
use zk_algorithm::reputation_system::*; // Using BN254 curve's scalar field

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Zero-Knowledge Time-Decaying Exposure Reputation Demo ===\n");

    // ========================================
    // SETUP: Initialize system
    // ========================================
    println!("🔧 Initializing system...");

    // Create genesis state (clean reputation)
    let initial_state = ExposureState::<Fr>::genesis(None);
    let commitment_randomness = Fr::from(12345u64);

    // Create prover
    let mut prover = ReputationProver::new(initial_state.clone(), commitment_randomness);

    // Create verifier
    let verifier = ReputationVerifier::new();

    let initial_commitment = StateCommitment::from_state(&initial_state, commitment_randomness);

    println!("✓ System initialized");
    println!("  Initial exposure: 0");
    println!("  Initial timestamp: 0\n");

    // ========================================
    // SCENARIO 1: User commits fraud → Penalty
    // ========================================
    println!("📍 Scenario 1: Fraud detected → Apply penalty");

    let penalty_amount = Fr::from(500u64);
    let penalty_input = TransitionInput::penalty(penalty_amount, 0);

    prover.apply_transition(penalty_input)?;

    println!("✓ Penalty applied: +500 exposure");
    #[cfg(test)]
    {
        println!("  Current exposure: {}", prover.get_current_exposure());
    }
    println!();

    // ========================================
    // SCENARIO 2: 30 days pass → Time decay
    // ========================================
    println!("📍 Scenario 2: 30 days pass → Natural decay");

    let time_delta = 30u64; // 30 time units (e.g., days)
    let decay_input = TransitionInput::time_decay(time_delta);

    prover.apply_transition(decay_input)?;

    println!("✓ Time decay applied: 30 days");
    println!("  Exposure decreased due to exponential decay\n");

    // ========================================
    // SCENARIO 3: More penalties
    // ========================================
    println!("📍 Scenario 3: Another violation → Penalty");

    let penalty_input_2 = TransitionInput::penalty(Fr::from(200u64), 5);

    prover.apply_transition(penalty_input_2)?;

    println!("✓ Penalty applied: +200 exposure (after 5 more days)");
    println!();

    // ========================================
    // SCENARIO 4: User completes recovery program
    // ========================================
    println!("📍 Scenario 4: Completed recovery program → Recovery");

    let recovery_amount = Fr::from(150u64);
    let recovery_input = TransitionInput::recovery(recovery_amount, 10);

    prover.apply_transition(recovery_input)?;

    println!("✓ Recovery applied: -150 exposure (after 10 more days)");
    println!();

    // ========================================
    // SCENARIO 5: More time passes
    // ========================================
    println!("📍 Scenario 5: 60 more days pass → More decay");

    let decay_input_2 = TransitionInput::time_decay(60);
    prover.apply_transition(decay_input_2)?;

    println!("✓ Significant decay: 60 days");
    println!();

    // ========================================
    // VERIFICATION: Prove exposure < threshold
    // ========================================
    println!("🔐 Generating threshold proof...");

    let threshold = Fr::from(400u64);
    let proof = prover.prove_threshold(threshold)?;

    println!("✓ Proof generated");
    println!("  Threshold: 400");
    println!(
        "  Claim: exposure < threshold? {}\n",
        proof.is_below_threshold
    );

    // ========================================
    // Verifier checks the proof
    // ========================================
    println!("✅ Verifying proof...");

    let is_valid = verifier.verify_threshold(&initial_commitment, &proof)?;

    if is_valid {
        println!("✓ Proof is VALID");

        if proof.is_below_threshold {
            println!("✓ User's exposure is below threshold");
            println!("  → User is approved for sensitive action");
        } else {
            println!("✗ User's exposure exceeds threshold");
            println!("  → User is denied for sensitive action");
        }
    } else {
        println!("✗ Proof is INVALID");
    }

    // ========================================
    // SCENARIO 6 (NEW): Policy-Driven Verification
    // ========================================
    println!("📍 Scenario 6: Checking against a higher threshold (Policy B)");

    // Demonstrate that the system allows policy-driven checks
    // The same user state can be verified against different thresholds
    let higher_threshold = Fr::from(700u64);
    let proof_high = prover.prove_threshold(higher_threshold)?;

    println!("🔐 Generating proof for Threshold: 700...");

    let is_valid_high = verifier.verify_threshold(&initial_commitment, &proof_high)?;

    if is_valid_high && proof_high.is_below_threshold {
        println!("✓ Proof is VALID");
        println!("✓ User's exposure is below threshold (700)");
        println!("  → User is APPROVED for Policy B (lower security clearance)");
    } else {
        println!("✗ User denied");
    }
    println!();

    // ========================================
    // TECHNICAL CLARIFICATIONS
    // ========================================
    println!("ℹ️  Technical Notes:");
    println!("  • Time Source: Modeled as monotonic counter (block height/epoch)");
    println!("  • Decay Model: Lookup-table approximation (λ ≈ 0.02/day)");
    println!("  • Disclosure:  Threshold-only (Infinite-State Machine approach)");
    println!("  • Security:    Monotonic exposure under penalties");
    println!("  • Anti-gaming: Recovery bounded per epoch (implicit)");
    println!("  • Folding:     Each transition is a valid IVC step");
    println!();

    // ========================================
    // PRIVACY GUARANTEE
    // ========================================
    println!("🔒 Privacy Analysis:");
    println!("  ✓ Verifier learns: exposure < 400? (boolean)");
    println!("  ✗ Verifier does NOT learn:");
    println!("    - Exact exposure value");
    println!("    - Number of violations");
    println!("    - Penalty amounts");
    println!("    - Recovery history");
    println!("    - Individual timestamps");
    println!();

    // ========================================
    // PERFORMANCE METRICS
    // ========================================
    println!("⚡ Performance Metrics:");

    let num_transitions = 5;
    println!("  Number of state transitions: {}", num_transitions);
    println!(
        "  Estimated proof size: {} bytes",
        utils::estimate_proof_size(num_transitions)
    );
    println!(
        "  Estimated prover time: {} ms",
        utils::estimate_prover_time(num_transitions)
    );
    println!(
        "  Estimated verifier time: {} ms",
        utils::estimate_verifier_time(num_transitions)
    );
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_flow() {
        // Run the main flow as a test
        main().expect("Demo should complete successfully");
    }
}
