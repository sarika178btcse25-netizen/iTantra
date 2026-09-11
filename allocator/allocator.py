"""
Bandwidth Allocator — aligned to the shared, frozen contract.

Updated per teammate review (this round):

1. `text` is now `str`, not `List[str]`. STT hands off the full utterance
   as a single string ("Send backup to grid reference 4729"), not a
   pre-split token list, and the backend already calls allocate() with
   that raw string. Tokenizing (`text.split()`) now happens inside
   allocate() so the function actually matches the real caller, instead
   of silently expecting an input shape nobody upstream produces.

2. `language` is now a REQUIRED parameter with NO default (previously
   defaulted to "en"). Team decision was Option A: language is officially
   part of the allocator's input contract, not an internal implementation
   detail. Removing the default is deliberate, not incidental — a silent
   "en" fallback is exactly the failure mode the team flagged: Hindi audio
   whose language tag gets dropped somewhere upstream would otherwise
   produce a Packet silently marked "en", and TTS would render English
   output on stage with no error anywhere in the pipeline. Failing loudly
   at the allocate() call site (a missing-argument TypeError, immediately,
   in testing) is strictly preferable to a silent wrong-language demo
   failure. This is a contract change and should be reflected in
   contracts/interfaces.md / schemas.py once the team confirms.

Everything else from the prior round is unchanged (already reviewed and
signed off as correct): shared Packet import, confidence-based priority
(criticality * (1 - confidence)), continuous protection values instead of
word-dropping, allocate_from_tagger() kept as an internal convenience
wrapper (tagger is still NOT a formal pipeline boundary — that's the one
remaining open item, separate from this round's fixes), and
reference_bitrate_kbps / protect_fraction kept as tunable implementation
details, not contract-level constants.
"""

from typing import List
from channel.packet import Packet


class BandwidthAllocator:
    def __init__(self, reference_bitrate_kbps: float = 8.0, min_protect_fraction: float = 0.15):
        # reference_bitrate_kbps: the bitrate at which we consider the
        # channel "good enough" to protect (almost) everything. Below
        # this, the fraction of words that get strong protection shrinks
        # proportionally. Tunable implementation detail — calibrate against
        # the real channel simulator's actual bitrate range, not a contract
        # value.
        self.reference_bitrate_kbps = reference_bitrate_kbps
        self.min_protect_fraction = min_protect_fraction

    def allocate(
        self,
        text: str,
        confidence: List[float],
        criticality: List[float],
        channel_bitrate_kbps: float,
        language: str,
    ) -> Packet:
        """
        Matches the frozen contract signature (language now included per
        team decision). `text` is the raw STT string; it's tokenized here
        by whitespace. confidence/criticality must have one entry per
        resulting token, in order.

        `language` has no default on purpose — see module docstring.
        Passing it explicitly at every call site is required; there is no
        "assume English" fallback anywhere in this function.
        """
        tokens = text.split()
        n = len(tokens)

        if not (n == len(confidence) == len(criticality)):
            raise ValueError(
                f"tokens ({n}), confidence ({len(confidence)}), and "
                f"criticality ({len(criticality)}) must be the same length — "
                f"they're parallel arrays over the same words. "
                f"tokens={tokens!r}"
            )
        if language not in {"en", "hi"}:
            raise ValueError(f"language must be 'en' or 'hi', got {language!r}")

        if n == 0:
            return Packet(
                tokens=[], confidence_per_token=[], criticality_per_token=[],
                protection_per_token=[], allocated_for_bitrate_kbps=channel_bitrate_kbps,
                language=language,
            )

        # Core USP formula: priority = criticality * (1 - confidence).
        # A critical word the STT was already confident about doesn't need
        # protection as urgently as a critical word it was unsure about.
        priority = [criticality[i] * (1 - confidence[i]) for i in range(n)]

        # How much of the sentence can we afford to protect strongly, given
        # the current bitrate? Scales linearly with bitrate, clamped to a
        # floor so SOMETHING always gets protected even on a dying channel.
        protect_fraction = max(
            self.min_protect_fraction,
            min(1.0, channel_bitrate_kbps / self.reference_bitrate_kbps)
        )
        num_protected = max(1, round(n * protect_fraction))

        # Rank by priority (ties broken toward earlier words)
        ranked_indices = sorted(
            range(n), key=lambda i: (priority[i], -i), reverse=True
        )
        protected_indices = set(ranked_indices[:num_protected])

        # Continuous protection value: protected words scale from 1.0 down
        # to a mid floor by rank; everyone else gets a low, non-zero
        # baseline (words are never deleted outright under this model,
        # just left more exposed).
        protection_per_token = [0.0] * n
        for rank, idx in enumerate(ranked_indices[:num_protected]):
            span = max(1, num_protected - 1)
            protection_per_token[idx] = 1.0 - 0.4 * (rank / span)
        for i in range(n):
            if i not in protected_indices:
                protection_per_token[i] = 0.1

        return Packet(
            tokens=tokens,
            confidence_per_token=list(confidence),
            criticality_per_token=list(criticality),
            protection_per_token=protection_per_token,
            allocated_for_bitrate_kbps=channel_bitrate_kbps,
            language=language,
        )

    def allocate_from_tagger(
        self,
        tagged_packet: dict,
        confidence_per_token: List[float],
        channel_bitrate_kbps: float,
    ) -> Packet:
        """
        Convenience wrapper for composing with tagger.py directly.
        tagged_packet is tagger.assign_criticality()'s output; caller
        supplies confidence_per_token from STT separately, aligned by
        index to tagged_packet["tokens"].

        Note: this still builds `text` back into a single string internally
        so it can go through the same allocate() path as everything else —
        one tokenization path, not two.
        """
        tagger_tokens = tagged_packet.get("tokens", [])
        if "language" not in tagged_packet or not tagged_packet["language"]:
            raise ValueError("tagged_packet is missing 'language' — must be passed through explicitly.")
        if len(confidence_per_token) != len(tagger_tokens):
            raise ValueError(
                f"confidence_per_token has {len(confidence_per_token)} entries but "
                f"tagged_packet has {len(tagger_tokens)} tokens — these must align 1:1."
            )
        words = [t["word"] for t in tagger_tokens]
        criticality = [t["criticality_score"] for t in tagger_tokens]
        text = " ".join(words)
        return self.allocate(
            text=text,
            confidence=confidence_per_token,
            criticality=criticality,
            channel_bitrate_kbps=channel_bitrate_kbps,
            language=tagged_packet["language"],
        )


if __name__ == "__main__":
    import tagger
    from channel.simulator import send

    allocator = BandwidthAllocator()

    print("=== TEST 1: allocate() returns the SHARED channel.packet.Packet ===")
    tagged = tagger.assign_criticality(
        "Um, listen to me Commander Sharma, basically we need 2 choppers at 28.5N and 77.1E immediately.",
        lang="en"
    )
    mock_confidence = [0.95] * len(tagged["tokens"])
    for i, t in enumerate(tagged["tokens"]):
        if t["word"] == "28.5N":
            mock_confidence[i] = 0.40

    packet = allocator.allocate_from_tagger(tagged, mock_confidence, channel_bitrate_kbps=3.0)
    assert isinstance(packet, Packet), "must return the shared channel.packet.Packet"
    print("type:", type(packet).__module__ + "." + type(packet).__name__)
    print("language:", packet.language, "(must be lowercase)")
    print("tokens:", packet.tokens)
    print("protection_per_token:", [round(p, 2) for p in packet.protection_per_token])

    print("\n=== TEST 2: verify criticality is EQUAL before comparing protection ===")
    idx_285n = packet.tokens.index("28.5N")
    idx_immediately = packet.tokens.index("immediately")
    crit_285n = packet.criticality_per_token[idx_285n]
    crit_immediately = packet.criticality_per_token[idx_immediately]
    print(f"  28.5N       criticality={crit_285n}  confidence={packet.confidence_per_token[idx_285n]}")
    print(f"  immediately criticality={crit_immediately}  confidence={packet.confidence_per_token[idx_immediately]}")
    assert crit_285n == crit_immediately, (
        "test doesn't isolate the confidence effect unless both words have "
        "equal criticality — check the tagger output before trusting this assert"
    )
    print(f"  28.5N       (confidence={packet.confidence_per_token[idx_285n]}) -> protection={packet.protection_per_token[idx_285n]:.2f}")
    print(f"  immediately (confidence={packet.confidence_per_token[idx_immediately]}) -> protection={packet.protection_per_token[idx_immediately]:.2f}")
    assert packet.protection_per_token[idx_285n] > packet.protection_per_token[idx_immediately], \
        "core USP broken: uncertain critical word should be protected MORE than a confident one, given equal criticality"
    print("  confirmed: with equal criticality, the less-confident word is protected more")

    print("\n=== TEST 3: Allocator -> Channel boundary works end to end ===")
    received_good = send(packet, bitrate_kbps=3.0, noise_level=0.3)
    print("Received (good channel run):", received_good)

    print("\n=== TEST 4: language format is lowercase, matching the frozen contract ===")
    assert packet.language in {"en", "hi"}, f"language must be lowercase en/hi, got {packet.language!r}"
    print("confirmed:", packet.language)

    print("\n=== TEST 5: text is now a plain string, tokenized internally ===")
    direct_packet = allocator.allocate(
        text="Send backup to grid reference 4729",
        confidence=[0.9, 0.85, 0.99, 0.3, 0.4, 0.6],
        criticality=[0.6, 0.9, 0.2, 0.9, 0.9, 0.5],
        channel_bitrate_kbps=4.0,
        language="en",
    )
    print("tokens (split from string):", direct_packet.tokens)
    assert direct_packet.tokens == "Send backup to grid reference 4729".split()
    print("confirmed: plain string input tokenized correctly")

    print("\n=== TEST 6: language has NO default — omitting it must fail loudly ===")
    try:
        allocator.allocate(
            text="test",
            confidence=[0.5],
            criticality=[0.5],
            channel_bitrate_kbps=4.0,
        )
        raise AssertionError("allocate() should have raised TypeError for missing language")
    except TypeError as e:
        print("confirmed: missing language raises TypeError ->", e)

    print("\nALL TESTS PASSED")