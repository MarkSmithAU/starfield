#version 330 core
in vec2 v_uv;
out vec4 f_color;

uniform samplerCube u_sky;
uniform mat3 u_camR;     // camera -> world
uniform vec2 u_tan;
uniform float u_bright;

void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec3 d = normalize(vec3(ndc * u_tan, 1.0));
    f_color = vec4(texture(u_sky, u_camR * d).rgb * u_bright, 1.0);
}
