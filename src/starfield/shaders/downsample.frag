#version 330 core
// 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of Duty")
in vec2 v_uv;
out vec4 f_color;
uniform sampler2D u_src;
uniform vec2 u_srcTexel;
uniform int u_first;

void main() {
    vec2 t = u_srcTexel;
    vec2 uv = v_uv;
    vec3 a = texture(u_src, uv + t * vec2(-2.0, 2.0)).rgb;
    vec3 b = texture(u_src, uv + t * vec2(0.0, 2.0)).rgb;
    vec3 c = texture(u_src, uv + t * vec2(2.0, 2.0)).rgb;
    vec3 d = texture(u_src, uv + t * vec2(-2.0, 0.0)).rgb;
    vec3 e = texture(u_src, uv).rgb;
    vec3 f = texture(u_src, uv + t * vec2(2.0, 0.0)).rgb;
    vec3 g = texture(u_src, uv + t * vec2(-2.0, -2.0)).rgb;
    vec3 h = texture(u_src, uv + t * vec2(0.0, -2.0)).rgb;
    vec3 i = texture(u_src, uv + t * vec2(2.0, -2.0)).rgb;
    vec3 j = texture(u_src, uv + t * vec2(-1.0, 1.0)).rgb;
    vec3 k = texture(u_src, uv + t * vec2(1.0, 1.0)).rgb;
    vec3 l = texture(u_src, uv + t * vec2(-1.0, -1.0)).rgb;
    vec3 m = texture(u_src, uv + t * vec2(1.0, -1.0)).rgb;
    vec3 r = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    if (u_first == 1) r = min(r, vec3(400.0));
    f_color = vec4(max(r, 0.0), 1.0);
}
