#version 330 core
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_tex;
uniform float u_alpha;

void main() {
    vec4 c = texture(u_tex, v_uv);
    f_color = vec4(c.rgb, c.a * u_alpha);
}
