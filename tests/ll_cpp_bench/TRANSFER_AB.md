# M4 transfer experiments

Target: EP8 BS32/48/64, with BS72 and BS256 retained. Both experiments use
`ready-snapshot`'s route planning, dependency scheduler and 28/28 core split.
The frozen `base` and existing Dispatch sources are unchanged.

## wqe-prep (ready=9)

The per-core staging address is known before the payload copy completes.
After queuing that copy and MTE3_S(7), construct a single HNS-1825 WRITE WQE
in an independent 128-byte UB area. The public SDK encoder supplies control,
task and data fields, including owner/wrap, remote MR translation and keys.
No valid SQ bytes or producer index are published yet. After the payload
wait, use the SDK's next-owner invalidation, UB-to-SQ copy, cache flush and
doorbell helper. Each peer/core still owns its original QP and generates one
WQE and one CQE per packet. The SDK doorbell updates its normal head mirror.

Admission uses min(SQ depth, CQ depth) and the SDK's ten-entry margin. A full
queue cannot poll CQ while the payload event is live; that packet falls back
to the unchanged SDK put after the payload wait. No queue reservation crosses
another submit, dependency wait, kernel or core. This adapter targets the
locked HNS-1825 implementation (`4c671684e1698aec199619850a3120bfdf85c95e`),
not XSCALE aggregate APIs. The SHMEM submodule and installed SDK are not edited.

The private UB is included in M4 workspace accounting and cannot overlap
inputs, outputs or the reduced `low_` buffer. Planning may borrow idle payload
space before use. The descriptor and doorbell copy finish before the next
packet or M5 reuses phase scratch. DIRECT cores do not enable preparation.

## mte-pack (ready=10)

Keep the final FP32-to-BF16/FP16 conversion and arithmetic order unchanged.
For remote partials, omit the two Vector copies that interleave 480-byte
payload blocks with 32-byte flags in the output UB. Instead issue:

1. Strided payload UB-to-GM copies for complete 480-byte blocks.
2. A final short payload copy if required.
3. Strided copies of the original initialized 32-byte flags into that GM packet.

UB source stride uses 32-byte units; GM destination stride uses bytes. The
packet length, every flag position and all valid hidden elements match the
original wire. Unused padding in the last payload block is not consumed.
The original V-to-MTE3 dependency is retained. The immediate MTE3_S wait
protects `low_` before a subsequent reduction can overwrite it. Only after
all payload and flag copies finish may the SDK borrow the packet's first
128 bytes as scratch and publish the RDMA write. Local returns use the old
packed output path. No additional UB or GM is allocated by this variant.

Both experiments preserve all 128 KiB/core GM slots, one quiet per touched
peer at a capacity boundary, and the final flush after M5. M5 never overwrites
the GM staging source. Two- and four-host topologies use the existing physical
rank grouping; there is no assumption of exactly one remote host.

## Read the experiment

Default build/run includes four clean and four matching profile variants.
Compare clean mean, P50 and min/max first. In diagnostics, compare M4 pack,
stage issue, remaining stage wait, put, first/last put, final flag wait and
quiet for the same rank/core/sample. `xfer` appends prepare ticks, prepared
packets, SDK fallbacks and direct MTE packets; full CSV/Perfetto retains them.
Preparation overlaps DMA, so its elapsed ticks are not additive network time.
Extra MTE commands or Scalar preparation may cost more than they save; hardware
benefit is unverified. Neither experiment reduces the number of network bytes
or WQEs.

CPU checks execute the production adapters under delayed DMA and NIC reads,
including partial tails, queue wrap/capacity fallback, 4/16 ranks per host,
two/four hosts and stage recycling. SDK stubs validate call/order/address
contracts, not actual NIC encoding or CANN compiler behavior. CANN build,
device correctness and latency still require the target machines.
