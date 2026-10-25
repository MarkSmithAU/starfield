#version 330 core
// Instanced star sprites drawn as motion-blurred capsules.
// u_infinite = 1: in_star.xyz is a direction (stars at infinity, only rotation moves them)
// u_infinite = 0: in_star.xyz is a position inside a wrapping box around the camera

in vec2 in_corner;
in vec4 in_star;    // xyz position/direction, w luminosity
in vec4 in_color;   // rgb colour, a: physical radius (3D) or sigma in px (infinite)

uniform int u_infinite;
uniform mat3 u_view;        // world -> camera
uniform vec3 u_campos;      // camera position modulo box size
uniform float u_L;
uniform vec3 u_vel;         // camera velocity, camera space
uniform vec3 u_angvel;      // camera angular velocity, camera space
uniform float u_shutter;
uniform vec2 u_focal;       // focal length in pixels
uniform vec2 u_res;
uniform float u_bright;
uniform float u_d0;
uniform float u_logfar;
uniform float u_doppler;
uniform float u_streakK;
uniform float u_pxScale;

noperspective out vec2 v_local;
flat out vec3 v_col;
flat out float v_len;
flat out float v_sig;

const float NEAR = 0.5;

void hide() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_col = vec3(0.0); v_local = vec2(0.0); v_len = 0.0; v_sig = 1.0;
}

void main() {
    vec3 p, vrel;
    float flux, sig, depth;
    if (u_infinite == 1) {
        p = u_view * in_star.xyz;
        vrel = -cross(u_angvel, p);
        flux = in_star.w * u_bright * u_pxScale * u_pxScale;
        sig = in_color.a * u_pxScale;
        depth = 1.0;
    } else {
        vec3 rel = mod(in_star.xyz - u_campos + 0.5 * u_L, u_L) - 0.5 * u_L;
        p = u_view * rel;
        float dist = length(p);
        float fade = smoothstep(0.5 * u_L, 0.36 * u_L, dist) * smoothstep(1.0, 6.0, dist);
        vrel = -(u_vel + cross(u_angvel, p));
        flux = in_star.w * u_bright * fade / (1.0 + pow(dist / u_d0, 1.6)) * u_pxScale * u_pxScale;
        sig = max(in_color.a * u_focal.y / max(p.z, NEAR), 0.8 * u_pxScale);
        depth = log2(1.0 + dist) / u_logfar;
    }
    if (p.z < NEAR || flux < 1e-4) { hide(); return; }

    // where the star was one shutter interval ago -> streak
    vec3 q = p - vrel * u_shutter;
    if (q.z < NEAR) {
        float t = (p.z - NEAR) / max(p.z - q.z, 1e-6);
        q = p + (q - p) * t;
    }
    vec2 b = p.xy / p.z * u_focal;
    vec2 a = q.xy / q.z * u_focal;
    vec2 ab = b - a;
    float len = length(ab);
    vec2 dir = len > 1e-3 ? ab / len : vec2(1.0, 0.0);
    len = min(len, 0.8 * u_res.y);
    a = b - dir * len;

    vec2 perp = vec2(-dir.y, dir.x);
    float ext = 3.2 * sig;
    float along = mix(-ext, len + ext, in_corner.x * 0.5 + 0.5);
    float across = in_corner.y * ext;
    vec2 pix = a + dir * along + perp * across;

    // spread energy over the streak (partially conserved so warp streaks stay vivid)
    float area = 6.2831853 * sig * sig + 2.5066283 * sig * len * u_streakK;
    vec3 col = in_color.rgb;
    float ahead = clamp(p.z / length(p), 0.0, 1.0);
    col = mix(col, col * vec3(0.55, 0.8, 1.7), u_doppler * ahead * ahead);

    v_col = col * flux / area;
    v_local = vec2(along, across);
    v_len = len;
    v_sig = sig;
    gl_Position = vec4(pix / (0.5 * u_res), depth * 2.0 - 1.0, 1.0);
}
