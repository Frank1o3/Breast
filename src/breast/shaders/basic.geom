#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 v_pos[];
in vec3 v_normal[];

out vec3 g_pos;
out vec3 g_normal;
out vec3 g_barycentric;

void main() {
    // Calculate face normal from triangle vertices
    vec3 edge1 = gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 edge2 = gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz;
    vec3 face_normal = normalize(cross(edge1, edge2));
    
    // Emit three vertices with barycentric coordinates
    for(int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        g_pos = v_pos[i];
        g_normal = face_normal;
        
        // Set barycentric coordinates (1,0,0), (0,1,0), (0,0,1)
        g_barycentric = vec3(0.0);
        g_barycentric[i] = 1.0;
        
        EmitVertex();
    }
    EndPrimitive();
}
