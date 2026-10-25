#version 330 core
// Bloom combine, lens flares, warp radial blur + chromatic aberration, ACES tonemap.
in vec2 v_uv;
out vec4 f_color;

uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform float u_bloomStrength;
uniform float u_exposure;
uniform float u_warp;
uniform float u_time;
uniform vec2 u_res;
uniform int u_nflare;
uniform vec4 u_flare[2];      // ndc xy, intensity, unused
uniform vec3 u_flareCol[2];

const int NG = 7;
const float GPOS[NG] = float[NG](-0.22, -0.45, -0.72, -1.05, -1.45, 0.38, 0.62);
const float GSIZE[NG] = float[NG](0.045, 0.09, 0.03, 0.14, 0.06, 0.025, 0.05);
const vec3 GCOL[NG] = vec3[NG](
    vec3(1.0, 0.6, 0.3), vec3(0.4, 0.8, 1.0), vec3(1.0, 0.4, 0.8), vec3(0.3, 0.5, 1.0),
    vec3(0.6, 1.0, 0.6), vec3(1.0, 0.8, 0.5), vec3(0.6, 0.6, 1.0));

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 sceneAt(vec2 uv) {
    vec2 c = uv - 0.5;
    float ca = 0.0007 + 0.006 * u_warp;
    if (u_warp < 0.01) {
        return vec3(texture(u_hdr, 0.5 + c * (1.0 + ca)).r,
                    texture(u_hdr, uv).g,
                    texture(u_hdr, 0.5 + c * (1.0 - ca)).b);
    }
    vec3 s = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 24; i++) {
        float k = (float(i) + hash12(gl_FragCoord.xy + float(i))) / 24.0;
        float sc = 1.0 - k * 0.05 * u_warp;
        float w = 1.0 - 0.6 * k;
        s += w * vec3(texture(u_hdr, 0.5 + c * sc * (1.0 + ca)).r,
                      texture(u_hdr, 0.5 + c * sc).g,
                      texture(u_hdr, 0.5 + c * sc * (1.0 - ca)).b);
        wsum += w;
    }
    return s / wsum;
}

vec3 flares(vec2 uv) {
    vec2 asp = vec2(u_res.x / u_res.y, 1.0);
    vec2 p = (uv - 0.5) * asp;
    vec3 sum = vec3(0.0);
    for (int j = 0; j < 2; j++) {
        if (j >= u_nflare) break;
        vec2 s = u_flare[j].xy * 0.5 * asp;
        float I = u_flare[j].z;
        vec3 col = u_flareCol[j];
        for (int k = 0; k < NG; k++) {
            vec2 g = s * GPOS[k];
            float rad = GSIZE[k];
            float d = length(p - g);
            float disc = smoothstep(rad, rad * 0.8, d) * (0.5 + 0.5 * smoothstep(0.0, rad, d));
            sum += GCOL[k] * col * disc * I * 0.018;
        }
        vec2 q = p - s;
        float d = length(q);
        // chromatic halo ring
        vec3 ring = vec3(exp(-pow((d - 0.43) * 30.0, 2.0)),
                         exp(-pow((d - 0.41) * 30.0, 2.0)),
                         exp(-pow((d - 0.39) * 30.0, 2.0)));
        sum += ring * col * I * 0.006;
        // anamorphic streak
        sum += col * vec3(0.45, 0.65, 1.0) * I * 0.25 * exp(-abs(q.y) * 700.0) * exp(-abs(q.x) * 2.5);
        // diffraction starburst
        float ang = atan(q.y, q.x);
        float spikes = pow(abs(cos(ang * 3.0 + 0.4)), 300.0) + 0.5 * pow(abs(cos(ang * 3.0 + 1.45)), 400.0);
        sum += col * I * spikes * exp(-d * 16.0) * 0.8;
        sum += col * I * exp(-d * 18.0) * 0.4;
    }
    return sum;
}

void main() {
    vec3 col = sceneAt(v_uv);
    vec3 bloom = texture(u_bloom, v_uv).rgb;
    col = mix(col, bloom, u_bloomStrength);
    col += flares(v_uv);

    // warp tunnel glow
    vec2 c = (v_uv - 0.5) * vec2(u_res.x / u_res.y, 1.0);
    col += vec3(0.25, 0.45, 1.0) * u_warp * 0.08 * exp(-dot(c, c) * 6.0);

    col *= u_exposure;
    col = aces(col);
    float vig = 1.0 - 0.28 * pow(length((v_uv - 0.5) * vec2(1.0, 0.8)), 2.2);
    col *= vig;
    col = pow(col, vec3(1.0 / 2.2));
    col += (hash12(gl_FragCoord.xy + fract(u_time) * 97.0) - 0.5) / 255.0;
    f_color = vec4(col, 1.0);
}
