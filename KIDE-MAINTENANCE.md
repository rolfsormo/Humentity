# Kide runtime maintenance branch

Base:ee5a1aa160b16735191daefebb1c1ec8743f904e. This branch preserves the
character-generation/CPU-animation API used by Kide's accepted native tester.
It targets Bevy0.20 and pins the compatible public OBJ dependency.

The later upstream standalone revision9b97be0 includes a useful0.20 port and
new shape/profile API, but its optional bevy_jolt dependency resolves an
inaccessible jolt_sys revision even in a default build. This runtime branch
therefore removes the unavailable Jolt feature, integration and examples.
Its original code remains in upstream Git history. No empty compatibility
feature or private workspace path is substituted.

Kide tracks future backend review in
https://github.com/rolfsormo/kide-mmo/blob/main/tickets/0128-character-physics-dependency-review.md .
Other inherited physics features are not claimed validated on Bevy0.20.
No ragdoll, physical-motor or GPU-crowd guarantee follows from the CPU studio.

Kide's studio, content, motion core and local paths are not runtime dependencies
of this library. Correctness patches should remain separable for future
upstream review; compatibility backports are not a request to duplicate
upstream's existing Bevy port.
