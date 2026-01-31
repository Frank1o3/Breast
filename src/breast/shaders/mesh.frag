#version 460 core

in vec3 g_view_pos;
in vec3 g_view_normal;
in vec3 g_barycentric;

out vec4 FragColor;

// ===== Lighting uniforms =====
uniform vec3 u_light_pos_view;
uniform vec3 u_light_color;

void main()
{
    vec3 N = normalize(g_view_normal);
    vec3 L = normalize(u_light_pos_view - g_view_pos);
    vec3 V = normalize(-g_view_pos);

    // Ambient lighting (so the cube is always visible)
    float ambient = 0.3;

    // Diffuse lighting
    float diff = max(dot(N, L), 0.0);

    // Combine lighting
    vec3 color = (ambient + diff) * u_light_color;

    // Optional: edge visualization using barycentrics
    float edge = min(min(g_barycentric.x, g_barycentric.y), g_barycentric.z);
    color *= smoothstep(0.0, 0.02, edge);

    FragColor = vec4(color, 1.0);
}
