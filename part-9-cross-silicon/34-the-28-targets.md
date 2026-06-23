# 34. Cross-silicon targets

> The compiler builds 28 architecture targets, one per silicon profile, under the fixed relation $M(n) \rightarrow H(n+12)$.
> A suffix letter selects the NE-core count, and the operation surface stops expanding at A15, so the surface measured on the M5 is the surface for everything above it.
> A device's runtime architecture string is a separate identifier from the compiler target name. A resolver-derived board-type sequence maps every shipping chip onto its generation.

The compiler that builds for the Apple Neural Engine has 28 architecture targets, one per silicon profile it knows how to construct.
Each target is a named hardware-abstraction-layer table the compiler builds by calling one per-architecture constructor, `ZinIrHal<T>::GetParams()`, and calling every constructor on a single host recovers the full set regardless of which chip runs it.

## Full set

[Table](#tbl:c34-targets) gives all 28 targets, each with its silicon class and decoded NE-core count.

| Target | Silicon and class | NE cores |
| --- | --- | --- |
| `H11`, `H12`, `M9`, `T0` | pre-A13 legacy | 1 to 4 |
| `H13` | A13, M1 base | 4 |
| `H13g` | M1 Pro, Max, Ultra | 8 |
| `T1` | A13 reference | 4 |
| `H14` | A14, M2 base | 4 |
| `H14g` | M2 Pro, Max | 8 |
| `H14c` | A14 Max-class | 32 |
| `H15` | A15, M3 base | 4 |
| `H15g` | M3 Pro, Max | 8 |
| `H15c` | A15 Max-class | 32 |
| `H16` | A16, M4 base | 4 |
| `H16g` | M4 Pro, Max | 8 |
| `H16s` | A16 Pro-class | 16 |
| `H16c` | A16 Max-class | 32 |
| `H17` | A17, M5 base | 4 |
| `H17a` | A17 variant | 4 |
| `H17g` | M5 Pro, Max | 8 |
| `H17s` | A17 Pro-class, the M5 | 16 |
| `H17c` | A17 Max-class | 32 |
| `H17d` | A17 Ultra-class | 64 |
| `H18` | A18 base | 4 |
| `M11` | small embedded ANE | 1 |
| `U1`, `U2`, `U3` | reference, not silicon | 4 |

Table: The 28 compiler targets, each with its silicon class and decoded NE-core count. {#tbl:c34-targets}

The names fall into four groups: the H-architecture targets that stand for shipping A-series and M-series silicon, the pre-A13 legacy targets, a single small embedded profile, and three reference targets that are not silicon at all.
A suffix letter selects the NE-core count within a generation, which the compiler decodes from the core-count field at hardware-abstraction-layer offset `0x238`.
The base name is 4 cores, the suffix `g` is 8, `s` is 16, `c` is 32, and `d` is 64, while `M9` and `M11` are single-core.
`H17s` is thus the 16-core Pro-class part that is the M5, and `H17d` is the 64-core Ultra-class die, the largest in the table.
These decoded `num_nes` values are the compiler's per-die core field, not Apple's marketing Neural Engine count; on the base M1 the decoded four stands against the published sixteen [AppleANE].

The reference targets hold placeholder limits that no part has: a maximum tensor depth of 1, a kernel-width limit of 1023, and no interchange-format support.
They are unconstrained validation profiles the compiler builds for its own checking, not addressable silicon.
The small embedded profile `M11` is addressable silicon.
It is an efficiency-class engine that has the A16-class feature flags but the A13-class 16384-dimension limit, a single NE core, and the odd kernel-width ceiling of 15 that is between the A13 value of 13 and the A14 value of 16.

## Capability tiers

[Table](#tbl:c34-tiers) groups the targets into capability tiers, giving each tier its dimension limit and the four gated capabilities that separate the generations.

| Tier | Targets | Max dimension | 3D conv | Texture engine | `sin`, `cos` | Dropout |
| --- | --- | --- | --- | --- | --- | --- |
| pre-A13 | `H11`, `H12`, `M9`, `T0` | 16384, depth 1 | no | no | no | no |
| A13 | `H13`, `H13g`, `T1` | 16384 | yes | no | no | no |
| A14 | `H14`, `H14g`, `H14c` | 16384 | yes | yes | no | no |
| A15 | `H15`, `H15g`, `H15c` | 16384 | yes | yes | yes | yes |
| A16 | `H16`, `H16g`, `H16s`, `H16c` | 65536 | yes | yes | yes | yes |
| A17 | `H17`, `H17a`, `H17g`, `H17s`, `H17c`, `H17d` | 65536 | yes | yes | yes | yes |
| A18 | `H18` | 65536 | yes | yes | yes | yes |
| small | `M11` | 16384 | yes | yes | yes | yes |
| reference | `U1`, `U2`, `U3` | 65535 placeholder | no | no | no | no |

Table: The capability tier of each target, with the dimension limit and the four gated capabilities that separate the generations. {#tbl:c34-tiers}

A17 and A18 add no operation over A16: identical dimension limits, identical kernel-width and kernel-depth ceilings, the same texture engine, same dropout and global-argmax flags, and same legal operation set.
They differ from A16 only in NE-core count, which scales throughput rather than legality.
The operation behavior measured on the M5, an H17 part, is thus the operation behavior of every target at or above A16, since the decoded capability tables are identical; the cross-silicon performance measurements of chapter 12 are predicted to carry to the unshipped generations on the same basis, with the per-chip rates confirmed only on the two measured silicon points.

## Silicon to target

The map from a shipping chip to its architecture is a resolver Apple distributes that decompiles cleanly.
The method `aneArchitectureType` on the private device-info class builds the architecture string from a board-type value read from the platform configuration store, switching on a strictly increasing board-type sequence.
The live anchor on an M1 Max reads board type 96, which resolves to `h13g` with a 16-core count, matching the registry exactly.
[Table](#tbl:c34-silicon) gives the resolver-derived map from system-on-chip to runtime architecture and compiler target across the M1 through M5 generations.

| Chip | Product | Runtime arch | Compiler target |
| --- | --- | --- | --- |
| T8103 | M1 base | `h13` | `H13` |
| T600x | M1 Pro, Max, Ultra | `h13g` | `H13G` |
| T8112 | M2 base | `h14` | `H14` |
| T602x | M2 Pro, Max | `h14g` | `H14G` |
| T8122 | M3 base | `h15` | `H15` |
| T603x | M3 Pro, Max | `h15g` | `H15G` |
| T8132 | M4 base | `h16` | `H16` |
| T604x | M4 Pro, Max | `h16g` | `H16G` |
| T8142 | M5 base | `h17` | `H17` |
| T605x | M5 Pro, Max | `h17s` | `H17s` |

Table: The resolver-derived map from system-on-chip to runtime architecture and compiler target, M1 through M5. {#tbl:c34-silicon}

The map follows the fixed $M(n) \rightarrow H(n+12)$ relation of chapter 12.
The sequence is anchored at both ends, the live M1 Max at `h13g` and the measured M5 Pro at `h17s`, and the intervening steps are corroborated independently.
A single shipping vision filter has exactly the five tables `H13`, `H14`, `H15`, `H16`, `H17`, the five Mac engine generations.
The board-type kext for an absent chip cannot be read on a different host, since only the running chip's table is resident, so each middle step rests on the anchored monotone sequence.

## Runtime string and compiler target

The architecture name a device reports at runtime is not the compiler target name.
The runtime string is the coarse form, `h1N` for a base part and `h1Ng` for a Pro, Max, or Ultra part, the only two variants the runtime emits on the desktop platform.
The compiler target is the finer set, the full `H17`, `H17s`, `H17c`, `H17d`, `H17g` family, of which the runtime collapses several onto one string.
A developer names the target by its compiler form and treats the runtime string as a separate identifier.

The direct compile entry point accepts any of the 28 target names and rejects an unknown name.
The dispatch library, in contrast, falls back silently when handed an unknown architecture, so a developer gates a cross-target compile against the known-name set before dispatching it.

## Interchange formats across the set

Each target has a per-chip table of accepted image-input formats, the interchange-format map at hardware-abstraction-layer offset `0x658`, keyed by a four-byte ASCII format tag.
[Table](#tbl:c34-formats) gives the accepted image-input format count by generation tier, with the format set each tier adds.

| Tier | Chips | Format count | Set |
| --- | --- | --- | --- |
| older, reference | `H11`, `H12`, `M9`, `T0`, `U1`, `U2`, `U3` | 0 | none |
| A13, M1 | `H13`, `H13g`, `T1` | 3 | `&BGA`, `&L0h`, `&L16` |
| A14 | `H14`, `H14g`, `H14c` | 13 | A13 set, RGBA-half, three compression variants |
| A15 and small | `H15`, `H15g`, `H15c`, `M11` | 16 | A14 set, YUV 4:2:0, luma-half |
| A16, A17, A18 | `H16`, `H17`, `H18` families | 14 | A15 set minus YUV 4:2:0 |

Table: The accepted image-input format count by generation tier, with the format set each tier adds. {#tbl:c34-formats}

The tag is a one-byte compression-variant prefix on a three-byte base pixel format.
The compiler does not parse the prefix character by character: it validates the whole four-byte tag against a 34-entry allow-list, and the prefix's meaning is the third byte of the format's packed-integer value, a packing-mode index on a uniform stride.
[Table](#tbl:c34-prefix) gives the compression-variant prefix on an interchange tag and the packing-mode index it selects.

| Prefix | Mode index | Meaning |
| --- | --- | --- |
| `&` | 0 | uncompressed, default raster surface |
| `-` | 1 | lossless compression, 32 by 32 macroblock |
| `/` | 2 | lossless compression, 16 by 16 macroblock |
| `\|` | 3 | lossless compression, mode 3 |
| `*` | 0 | compound prefix that sets the dynamic-channel flag |

Table: The compression-variant prefix on an interchange tag and the packing-mode index it selects. {#tbl:c34-prefix}

The packed integer that names each format is three bytes: a pixel class, a base-format code, and the packing-mode index.
The base-format codes are BGRA8 (`BGA`, code `0x11`), RGBA-half (`RhA`, code `0x13`), 8-bit luma (`L0h`, code `0x07`), 16-bit luma (`L16`, code `0x08`), and YUV 4:2:0 (`8f0` and `8v0`, code `0x09`).
A base format routes to a vector of 20-byte plane descriptors, each a tuple of width divisor, height divisor, element type, channel count, and depth.
BGRA8 is thus one four-channel uint8 plane, and YUV 4:2:0 is a luma plane with a half-resolution two-channel chroma plane.
The binary string that reads "Architecture only supports lossless compression" confirms that `&` is the uncompressed variant and that `-`, `/`, and `|` are the lossless-compressed packing families.

The A15 generation and the small embedded profile are the only targets in the set that accept YUV 4:2:0 input, in both full-range (`8f0`) and video-range (`8v0`) form.
The M5 and every A16-and-later part keep luma-half but drop the two YUV 4:2:0 formats.
The full per-target format records, the wider 10-bit and packed YUV family, and the plane-layout structures are appendix material.
