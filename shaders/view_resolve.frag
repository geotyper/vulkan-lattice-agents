#version 450

// Resolves the two transparency buffers onto the picture. `accumulation` holds
// the weighted sum of colour and alpha; `revealage` holds the product of every
// fragment's (1 - alpha), which is how much of the background still shows.
layout(set = 0, binding = 0) uniform sampler2D accumulation;
layout(set = 0, binding = 1) uniform sampler2D revealage;

layout(location = 0) in vec2 fragUv;
layout(location = 0) out vec4 outColour;

void main() {
    const vec4 accumulated = texture(accumulation, fragUv);
    const float revealed = texture(revealage, fragUv).r;
    // The divide is the average the weights were for. Guarded because a pixel
    // no fragment reached has a weight sum of zero, and the blend below drops it
    // anyway through an alpha of zero.
    const vec3 colour = accumulated.rgb / max(accumulated.a, 1.0e-5);
    outColour = vec4(colour, 1.0 - revealed);
}
