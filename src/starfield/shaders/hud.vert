#version 330 core
in vec2 in_pos;
uniform vec4 u_rect;   // x, y, w, h in pixels (origin top-left)
uniform vec2 u_res;
out vec2 v_uv;

void main() {
    vec2 px = u_rect.xy + in_pos * u_rect.zw;
    v_uv = in_pos;
    gl_Position = vec4(px.x / u_res.x * 2.0 - 1.0, 1.0 - px.y / u_res.y * 2.0, 0.0, 1.0);
}
