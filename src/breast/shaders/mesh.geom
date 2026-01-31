#version 460 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

// ===== Inputs from Vertex Shader =====
in VS_OUT {
    vec3 view_pos;
} gs_in[];

// ===== Outputs to Fragment Shader =====
out vec3 g_view_pos;
out vec3 g_view_normal;
out vec3 g_barycentric;

void main()
{
    // Triangle edges (view space)
    vec3 e0 = gs_in[1].view_pos - gs_in[0].view_pos;
    vec3 e1 = gs_in[2].view_pos - gs_in[0].view_pos;

    // Face normal (view space)
    vec3 face_normal = normalize(cross(e0, e1));

    // Back-face culling
    // Camera looks down -Z in view space
    if (face_normal.z <= 0.0)
        return;

    // Emit triangle with barycentric coords
    for (int i = 0; i < 3; i++)
    {
        g_view_pos    = gs_in[i].view_pos;
        g_view_normal = face_normal;

        g_barycentric = vec3(0.0);
        g_barycentric[i] = 1.0;

        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }

    EndPrimitive();
}
