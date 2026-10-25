#version 330 core
//#include common

// Renders one face of the background cubemap: Milky Way band, dust lanes,
// nebulae and a dense layer of unresolved faint stars.

in vec2 v_uv;
out vec4 f_color;

uniform int u_face;
uniform vec3 u_galN;    // galactic plane normal
uniform vec3 u_galC;    // direction of galactic centre (in plane)
uniform float u_seed;
uniform vec3 u_nebA;
uniform vec3 u_nebB;
uniform vec3 u_nebC;
uniform float u_faceRes;

vec3 faceDir(int face, vec2 st) {
    if (face == 0) return vec3(1.0, -st.y, -st.x);
    if (face == 1) return vec3(-1.0, -st.y, st.x);
    if (face == 2) return vec3(st.x, 1.0, st.y);
    if (face == 3) return vec3(st.x, -1.0, -st.y);
    if (face == 4) return vec3(st.x, -st.y, 1.0);
    return vec3(-st.x, -st.y, -1.0);
}

float starDust(vec3 d, float scale, float prob, float salt) {
    vec3 p = d * scale;
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 h = hash33(i + salt + u_seed);
    if (h.x > prob) return 0.0;
    vec3 c = 0.25 + 0.5 * hash33(i * 1.37 + 11.0 + salt);
    float r = length(f - c) / scale;              // distance in direction units
    float sigma = 0.9 / u_faceRes;                // about one texel
    float bright = pow(h.y, 5.0) * 1.2 + 0.05 * h.z;
    return bright * exp(-r * r / (2.0 * sigma * sigma));
}

void main() {
    vec2 st = v_uv * 2.0 - 1.0;
    vec3 d = normalize(faceDir(u_face, st));

    vec3 n = u_galN;
    vec3 e1 = u_galC;
    float lat = dot(d, n);
    float toCentre = dot(normalize(d - lat * n + 1e-6), e1);
    float bulge = exp(-pow(acos(clamp(dot(d, e1), -1.0, 1.0)), 2.0) / 0.30);
    float width = 0.09 + 0.07 * (toCentre * 0.5 + 0.5);
    float band = exp(-lat * lat / (width * width));

    // Milky Way: unresolved star light
    float n1 = fbm(d * 4.0 + u_seed, 6) * 0.5 + 0.5;
    float n2 = fbm(d * 13.0 + u_seed * 1.7, 5) * 0.5 + 0.5;
    float mw = band * (0.25 + 0.75 * n1) * (0.6 + 0.4 * n2) + 0.9 * bulge * exp(-lat * lat / 0.03);

    // dark dust lanes hugging the plane
    float dl = smoothstep(0.42, 0.68, fbm(d * 6.0 + u_seed * 2.3, 7) * 0.5 + 0.5);
    float lane = exp(-lat * lat / (0.30 * width * width + 0.0015));
    mw *= 1.0 - 0.88 * dl * lane;

    vec3 mwCol = mix(vec3(0.55, 0.62, 0.85), vec3(1.0, 0.80, 0.58), clamp(bulge * 0.9 + 0.25 * n2, 0.0, 1.0));
    vec3 col = mwCol * mw * 0.030;

    // nebulae: domain-warped fbm, masked into a few regions of the sky
    vec3 q = d * 1.6 + u_seed * 0.37;
    vec3 w = vec3(fbm(q, 4), fbm(q + vec3(5.2, 1.3, 2.8), 4), fbm(q + vec3(1.7, 9.2, 4.1), 4));
    float nb = fbm(q * 2.3 + 1.8 * w, 7) * 0.5 + 0.5;
    float mask = smoothstep(0.50, 0.78, fbm(d * 1.1 + u_seed * 0.71 + 20.0, 4) * 0.5 + 0.5);
    float dens = pow(clamp(nb, 0.0, 1.0), 3.2) * mask;
    vec3 nebCol = mix(u_nebA, u_nebB, smoothstep(0.25, 0.75, w.x * 0.5 + 0.5));
    nebCol = mix(nebCol, u_nebC, smoothstep(0.62, 0.9, nb) * 0.7);
    // dark filaments cut through the gas
    float fil = smoothstep(0.55, 0.75, fbm(q * 5.0 + 3.0 * w + 7.0, 5) * 0.5 + 0.5);
    col += nebCol * dens * (1.0 - 0.7 * fil) * 0.11;

    // unresolved faint stars, denser along the band
    float sd = starDust(d, 300.0, 0.30, 0.0) * (0.35 + 1.6 * band)
             + 0.6 * starDust(d, 620.0, 0.22, 17.0) * (0.25 + 2.2 * band);
    col += vec3(0.88, 0.92, 1.0) * sd * 0.22;

    col += vec3(0.0010, 0.0012, 0.0022);
    f_color = vec4(col, 1.0);
}
