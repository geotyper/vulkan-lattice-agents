#version 450

// One triangle covering the screen, from the vertex index alone. A quad would
// need two and a seam between them.
layout(location = 0) out vec2 fragUv;

void main() {
    fragUv = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(fragUv * 2.0 - 1.0, 0.0, 1.0);
}
