#version 330 core
// 3x3 tent upsample, additively blended into the next larger mip
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_texel;

void main() {
    vec2 t = u_texel;
    vec3 s = texture(u_src, v_uv + t * vec2(-1.0, 1.0)).rgb
           + texture(u_src, v_uv + t * vec2(0.0, 1.0)).rgb * 2.0
           + texture(u_src, v_uv + t * vec2(1.0, 1.0)).rgb
           + texture(u_src, v_uv + t * vec2(-1.0, 0.0)).rgb * 2.0
           + texture(u_src, v_uv).rgb * 4.0
           + texture(u_src, v_uv + t * vec2(1.0, 0.0)).rgb * 2.0
           + texture(u_src, v_uv + t * vec2(-1.0, -1.0)).rgb
           + texture(u_src, v_uv + t * vec2(0.0, -1.0)).rgb * 2.0
           + texture(u_src, v_uv + t * vec2(1.0, -1.0)).rgb;
    f_color = vec4(s / 16.0, 1.0);
}
