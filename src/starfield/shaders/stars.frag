#version 330 core
noperspective in vec2 v_local;
flat in vec3 v_col;
flat in float v_len;
flat in float v_sig;
out vec4 f_color;

void main() {
    float dx = v_local.x - clamp(v_local.x, 0.0, v_len);
    float d2 = dx * dx + v_local.y * v_local.y;
    float i = exp(-d2 / (2.0 * v_sig * v_sig));
    f_color = vec4(v_col * i, 0.0);
}
