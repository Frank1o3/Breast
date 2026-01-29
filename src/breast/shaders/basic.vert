#version 330
in vec3 in_vert;
out vec3 v_pos;
out vec3 v_normal;
uniform mat4 mvp;
uniform mat4 model;

void main() {
    gl_Position = mvp * vec4(in_vert, 1.0);
    v_pos = (model * vec4(in_vert, 1.0)).xyz;
    
    // Normal will be calculated in geometry shader or we'll compute it in fragment shader
    // For now, pass position for per-pixel normal calculation
    v_normal = vec3(0.0);
}
