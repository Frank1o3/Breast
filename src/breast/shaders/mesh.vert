#version 460 core

// ===== Inputs =====
layout (location = 0) in vec3 in_position; // WORLD-space position (physics-updated)

// ===== Uniforms =====
uniform mat4 u_view;
uniform mat4 u_proj;

// ===== Outputs to Geometry Shader =====
out VS_OUT {
    vec3 view_pos;
} vs_out;

void main()
{
    // World → View
    vec4 view_pos4 = u_view * vec4(in_position, 1.0);
    vs_out.view_pos = view_pos4.xyz;

    // View → Clip (GPU does projection)
    gl_Position = u_proj * view_pos4;
}
