#version 330
in vec3 g_pos;
in vec3 g_normal;
in vec3 g_barycentric;

out vec4 f_color;

uniform vec4 base_color;
uniform vec3 light_pos;
uniform int wireframe_mode;  // 0 = filled, 1 = wireframe, 2 = filled with edges
uniform float wireframe_width;

float edgeFactor() {
    // Calculate distance to nearest edge using barycentric coordinates
    vec3 d = fwidth(g_barycentric);
    vec3 a3 = smoothstep(vec3(0.0), d * wireframe_width, g_barycentric);
    return min(min(a3.x, a3.y), a3.z);
}

void main() {
    vec3 color;
    
    if (wireframe_mode == 1) {
        // Pure wireframe mode
        float edge = edgeFactor();
        if (edge > 0.99) {
            discard;
        }
        color = vec3(0.9, 0.9, 0.9);
    } else {
        // Filled mode with lighting
        vec3 norm = normalize(g_normal);
        vec3 light_dir = normalize(light_pos - g_pos);
        
        // Ambient
        float ambient_strength = 0.3;
        vec3 ambient = ambient_strength * base_color.rgb;
        
        // Diffuse
        float diff = max(dot(norm, light_dir), 0.0);
        vec3 diffuse = diff * base_color.rgb;
        
        // Specular
        vec3 view_dir = normalize(-g_pos);
        vec3 reflect_dir = reflect(-light_dir, norm);
        float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
        vec3 specular = 0.5 * spec * vec3(1.0, 1.0, 1.0);
        
        color = ambient + diffuse + specular;
        
        // Overlay wireframe in mode 2
        if (wireframe_mode == 2) {
            float edge = edgeFactor();
            color = mix(vec3(0.1, 0.1, 0.1), color, edge);
        }
    }
    
    f_color = vec4(color, base_color.a);
}
